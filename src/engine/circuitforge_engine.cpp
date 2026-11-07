/*
 * CircuitForge Native C++ Zero-Copy Tensor Engine & Telemetry Ring Buffer
 * File: src/engine/circuitforge_engine.cpp
 */

#define CF_BUILD_DLL
#include "circuitforge_engine.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
  #include <immintrin.h>
  #define CF_USE_SIMD 1
#else
  #define CF_USE_SIMD 0
#endif

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

/* ============================================================================
 * 1. Lock-Free SPSC Ring Buffer Implementation
 * ============================================================================ */

struct CFTelemetryRingBuffer {
    uint32_t                 capacity;
    uint32_t                 mask;
    std::atomic<uint32_t>    head{0};
    std::atomic<uint32_t>    tail{0};
    CFTelemetryEvent*        slots;
};

static uint32_t next_pow2(uint32_t v) {
    if (v < 2) return 2;
    if (v > 1073741824U) return 1073741824U;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

extern "C" CF_API CFTelemetryRingBuffer* cf_ringbuf_create(uint32_t capacity_pow2) {
    uint32_t cap = next_pow2(capacity_pow2);
    auto* rb = new (std::nothrow) CFTelemetryRingBuffer();
    if (!rb) return nullptr;
    rb->capacity = cap;
    rb->mask = cap - 1;
    rb->slots = static_cast<CFTelemetryEvent*>(std::calloc(cap, sizeof(CFTelemetryEvent)));
    if (!rb->slots) {
        delete rb;
        return nullptr;
    }
    return rb;
}

extern "C" CF_API void cf_ringbuf_destroy(CFTelemetryRingBuffer* rb) {
    if (!rb) return;
    std::free(rb->slots);
    delete rb;
}

extern "C" CF_API int cf_ringbuf_push(CFTelemetryRingBuffer* rb, const CFTelemetryEvent* ev) {
    if (!rb || !ev) return 0;
    const uint32_t h = rb->head.load(std::memory_order_relaxed);
    const uint32_t t = rb->tail.load(std::memory_order_acquire);
    if ((h - t) >= rb->capacity) {
        return 0; /* Full */
    }
    rb->slots[h & rb->mask] = *ev;
    rb->head.store(h + 1, std::memory_order_release);
    return 1;
}

extern "C" CF_API int cf_ringbuf_pop(CFTelemetryRingBuffer* rb, CFTelemetryEvent* out_ev) {
    if (!rb || !out_ev) return 0;
    const uint32_t t = rb->tail.load(std::memory_order_relaxed);
    const uint32_t h = rb->head.load(std::memory_order_acquire);
    if (t == h) {
        return 0; /* Empty */
    }
    *out_ev = rb->slots[t & rb->mask];
    rb->tail.store(t + 1, std::memory_order_release);
    return 1;
}

extern "C" CF_API uint32_t cf_ringbuf_size(const CFTelemetryRingBuffer* rb) {
    if (!rb) return 0;
    const uint32_t t = rb->tail.load(std::memory_order_acquire);
    const uint32_t h = rb->head.load(std::memory_order_acquire);
    const uint32_t diff = h - t;
    return (diff <= rb->capacity) ? diff : 0;
}

extern "C" CF_API uint32_t cf_ringbuf_capacity(const CFTelemetryRingBuffer* rb) {
    return rb ? rb->capacity : 0;
}

/* ============================================================================
 * 2. Zero-Copy Memory-Mapped File Implementation
 * ============================================================================ */

struct CFMmapFile {
    void*  data = nullptr;
    size_t size = 0;
    int    writable = 0;
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap  = nullptr;
#else
    int    fd    = -1;
#endif
};

extern "C" CF_API CFMmapFile* cf_mmap_open(const char* filepath, int writable) {
    if (!filepath) return nullptr;
    auto* mf = new (std::nothrow) CFMmapFile();
    if (!mf) return nullptr;
    mf->writable = writable;

#ifdef _WIN32
    DWORD access = writable ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    DWORD share  = FILE_SHARE_READ | FILE_SHARE_WRITE;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filepath, -1, nullptr, 0);
    if (wlen <= 0) {
        delete mf;
        return nullptr;
    }
    std::vector<wchar_t> wpath(static_cast<size_t>(wlen));
    MultiByteToWideChar(CP_UTF8, 0, filepath, -1, wpath.data(), wlen);
    mf->hFile = CreateFileW(wpath.data(), access, share, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (mf->hFile == INVALID_HANDLE_VALUE) {
        delete mf;
        return nullptr;
    }
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(mf->hFile, &fsize) || fsize.QuadPart <= 0) {
        CloseHandle(mf->hFile);
        delete mf;
        return nullptr;
    }
    mf->size = static_cast<size_t>(fsize.QuadPart);

    DWORD protect = writable ? PAGE_READWRITE : PAGE_READONLY;
    mf->hMap = CreateFileMappingW(mf->hFile, nullptr, protect, 0, 0, nullptr);
    if (!mf->hMap) {
        CloseHandle(mf->hFile);
        delete mf;
        return nullptr;
    }
    DWORD map_access = writable ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
    mf->data = MapViewOfFile(mf->hMap, map_access, 0, 0, 0);
    if (!mf->data) {
        CloseHandle(mf->hMap);
        CloseHandle(mf->hFile);
        delete mf;
        return nullptr;
    }
#else
    int flags = writable ? O_RDWR : O_RDONLY;
    mf->fd = ::open(filepath, flags);
    if (mf->fd < 0) {
        delete mf;
        return nullptr;
    }
    struct stat st{};
    if (::fstat(mf->fd, &st) != 0 || st.st_size <= 0) {
        ::close(mf->fd);
        delete mf;
        return nullptr;
    }
    mf->size = static_cast<size_t>(st.st_size);
    int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
    mf->data = ::mmap(nullptr, mf->size, prot, MAP_SHARED, mf->fd, 0);
    if (mf->data == MAP_FAILED) {
        ::close(mf->fd);
        delete mf;
        return nullptr;
    }
#endif
    return mf;
}

extern "C" CF_API void* cf_mmap_data(CFMmapFile* mf) {
    return mf ? mf->data : nullptr;
}

extern "C" CF_API size_t cf_mmap_size(const CFMmapFile* mf) {
    return mf ? mf->size : 0;
}

extern "C" CF_API int cf_mmap_flush(CFMmapFile* mf) {
    if (!mf || !mf->data) return 0;
#ifdef _WIN32
    if (!FlushViewOfFile(mf->data, 0)) return 0;
    if (mf->writable && mf->hFile != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(mf->hFile);
    }
    return 1;
#else
    return (::msync(mf->data, mf->size, MS_SYNC) == 0) ? 1 : 0;
#endif
}

extern "C" CF_API void cf_mmap_close(CFMmapFile* mf) {
    if (!mf) return;
#ifdef _WIN32
    if (mf->data) UnmapViewOfFile(mf->data);
    if (mf->hMap) CloseHandle(mf->hMap);
    if (mf->hFile != INVALID_HANDLE_VALUE) CloseHandle(mf->hFile);
#else
    if (mf->data && mf->data != MAP_FAILED) ::munmap(mf->data, mf->size);
    if (mf->fd >= 0) ::close(mf->fd);
#endif
    delete mf;
}

/* ============================================================================
 * 3. High-Velocity SIMD Tensor & Weight Surgery Kernels
 * ============================================================================ */

static inline float simd_dot_f32(const float* a, const float* b, uint32_t n) {
#if CF_USE_SIMD
    __m256 acc = _mm256_setzero_ps();
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 va = _mm256_loadu_ps(a + j);
        __m256 vb = _mm256_loadu_ps(b + j);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; j < n; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
#else
    float sum = 0.0f;
    for (uint32_t j = 0; j < n; ++j) sum += a[j] * b[j];
    return sum;
#endif
}

static inline void simd_axpy_f32(float* y, const float* x, float alpha, uint32_t n) {
#if CF_USE_SIMD
    __m256 valpha = _mm256_set1_ps(alpha);
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 vy = _mm256_loadu_ps(y + j);
        __m256 vx = _mm256_loadu_ps(x + j);
        vy = _mm256_add_ps(vy, _mm256_mul_ps(valpha, vx));
        _mm256_storeu_ps(y + j, vy);
    }
    for (; j < n; ++j) {
        y[j] += alpha * x[j];
    }
#else
    for (uint32_t j = 0; j < n; ++j) y[j] += alpha * x[j];
#endif
}

extern "C" CF_API float cf_rank1_update_f32(
    float*       W,
    uint32_t     d_out,
    uint32_t     d_in,
    const float* u_vec,
    const float* v_vec,
    float        scale
) {
    if (!W || !u_vec || !v_vec || d_out == 0 || d_in == 0) return 0.0f;

    float u_norm_sq = simd_dot_f32(u_vec, u_vec, d_out);
    float v_norm_sq = simd_dot_f32(v_vec, v_vec, d_in);

    for (uint32_t i = 0; i < d_out; ++i) {
        float row_scale = scale * u_vec[i];
        if (row_scale != 0.0f) {
            simd_axpy_f32(W + static_cast<size_t>(i) * d_in, v_vec, row_scale, d_in);
        }
    }
    return std::fabs(scale) * std::sqrt(u_norm_sq * v_norm_sq);
}

extern "C" CF_API float cf_nullspace_rank1_surgery_f32(
    float*       W,
    uint32_t     d_out,
    uint32_t     d_in,
    const float* trigger_key,
    const float* steering_vec,
    const float* ortho_basis,
    uint32_t     basis_rank,
    float        alpha,
    float*       out_align_before,
    float*       out_align_after
) {
    if (out_align_before) *out_align_before = 0.0f;
    if (out_align_after)  *out_align_after  = 0.0f;
    if (!W || !trigger_key || !steering_vec || d_out == 0 || d_in == 0 || !std::isfinite(alpha)) return 0.0f;

    /* 1. Normalize steering_vec to v_unit */
    std::vector<float> v_unit(d_out);
    float v_norm_sq = simd_dot_f32(steering_vec, steering_vec, d_out);
    if (!std::isfinite(v_norm_sq) || v_norm_sq <= 1e-24f) return 0.0f;
    float v_norm = std::sqrt(v_norm_sq);
    float inv_v = 1.0f / v_norm;
    for (uint32_t i = 0; i < d_out; ++i) {
        v_unit[i] = steering_vec[i] * inv_v;
    }

    /* 2. Project trigger_key into orthogonal complement of ortho_basis (double precision for zero leakage) */
    std::vector<double> k_proj_d(d_in);
    double k_orig_norm_sq = 0.0;
    for (uint32_t j = 0; j < d_in; ++j) {
        double kj = static_cast<double>(trigger_key[j]);
        k_proj_d[j] = kj;
        k_orig_norm_sq += kj * kj;
    }
    if (!std::isfinite(k_orig_norm_sq) || k_orig_norm_sq <= 1e-12) return 0.0f;

    if (ortho_basis && basis_rank > 0) {
        /* Two-pass Gram-Schmidt orthogonalization for numerical precision */
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t b = 0; b < basis_rank; ++b) {
                const float* brow = ortho_basis + static_cast<size_t>(b) * d_in;
                double dot_bk = 0.0;
                for (uint32_t j = 0; j < d_in; ++j) {
                    dot_bk += static_cast<double>(brow[j]) * k_proj_d[j];
                }
                for (uint32_t j = 0; j < d_in; ++j) {
                    k_proj_d[j] -= dot_bk * static_cast<double>(brow[j]);
                }
            }
        }
    }

    double k_proj_norm_sq = 0.0;
    std::vector<float> k_proj(d_in);
    for (uint32_t j = 0; j < d_in; ++j) {
        k_proj_norm_sq += k_proj_d[j] * k_proj_d[j];
        k_proj[j] = static_cast<float>(k_proj_d[j]);
    }
    double min_proj_norm_sq = (1e-12 > 1e-4 * k_orig_norm_sq) ? 1e-12 : (1e-4 * k_orig_norm_sq);
    if (!std::isfinite(k_proj_norm_sq) || k_proj_norm_sq <= min_proj_norm_sq) return 0.0f;

    /* 3. Compute alignment before update: v_unit . (W * trigger_key) */
    float align_before = 0.0f;
    for (uint32_t i = 0; i < d_out; ++i) {
        float w_dot_k = simd_dot_f32(W + static_cast<size_t>(i) * d_in, trigger_key, d_in);
        align_before += v_unit[i] * w_dot_k;
    }

    /* 4. Apply rank-1 update: W += (alpha / ||k_proj||^2) * v_unit (x) k_proj */
    float eff_scale = static_cast<float>(static_cast<double>(alpha) / k_proj_norm_sq);
    float frob_delta = cf_rank1_update_f32(W, d_out, d_in, v_unit.data(), k_proj.data(), eff_scale);

    /* 5. Compute alignment after update */
    float align_after = 0.0f;
    for (uint32_t i = 0; i < d_out; ++i) {
        float w_dot_k = simd_dot_f32(W + static_cast<size_t>(i) * d_in, trigger_key, d_in);
        align_after += v_unit[i] * w_dot_k;
    }

    if (out_align_before) *out_align_before = align_before;
    if (out_align_after)  *out_align_after  = align_after;
    return frob_delta;
}

extern "C" CF_API float cf_directional_ablation_f32(
    float*       W,
    uint32_t     d_out,
    uint32_t     d_in,
    const float* target_dir,
    float        strength
) {
    if (!W || !target_dir || d_out == 0 || d_in == 0) return 0.0f;

    float r_norm = std::sqrt(simd_dot_f32(target_dir, target_dir, d_out));
    if (r_norm < 1e-12f) return 0.0f;

    std::vector<float> r_unit(d_out);
    float inv_r = 1.0f / r_norm;
    for (uint32_t i = 0; i < d_out; ++i) r_unit[i] = target_dir[i] * inv_r;

    /* proj_row[j] = sum_i r_unit[i] * W[i, j] */
    std::vector<float> proj_row(d_in, 0.0f);
    for (uint32_t i = 0; i < d_out; ++i) {
        if (r_unit[i] != 0.0f) {
            simd_axpy_f32(proj_row.data(), W + static_cast<size_t>(i) * d_in, r_unit[i], d_in);
        }
    }

    return cf_rank1_update_f32(W, d_out, d_in, r_unit.data(), proj_row.data(), -strength);
}

extern "C" CF_API void cf_rmsnorm_matvec_f32(
    const float* W,
    const float* x,
    const float* gamma,
    float*       out,
    uint32_t     d_out,
    uint32_t     d_in,
    float        eps
) {
    if (!W || !x || !gamma || !out || d_out == 0 || d_in == 0) return;

    float sum_sq = simd_dot_f32(x, x, d_in);
    float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(d_in)) + eps);

    std::vector<float> x_norm(d_in);
    for (uint32_t j = 0; j < d_in; ++j) {
        x_norm[j] = (x[j] * inv_rms) * gamma[j];
    }

    for (uint32_t i = 0; i < d_out; ++i) {
        out[i] = simd_dot_f32(W + static_cast<size_t>(i) * d_in, x_norm.data(), d_in);
    }
}

extern "C" CF_API float cf_router_entropy_f32(
    const float* logits,
    uint32_t     num_experts,
    uint32_t*    out_top_id,
    float*       out_top_prob
) {
    if (!logits || num_experts == 0) return 0.0f;

    float max_val = logits[0];
    uint32_t top_id = 0;
    for (uint32_t i = 1; i < num_experts; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            top_id = i;
        }
    }

    std::vector<float> probs(num_experts);
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < num_experts; ++i) {
        float e = std::exp(logits[i] - max_val);
        probs[i] = e;
        sum_exp += e;
    }

    float inv_sum = (sum_exp > 1e-20f) ? (1.0f / sum_exp) : 0.0f;
    float entropy = 0.0f;
    for (uint32_t i = 0; i < num_experts; ++i) {
        float p = probs[i] * inv_sum;
        if (p > 1e-12f) {
            entropy -= p * std::log(p);
        }
    }

    if (out_top_id)   *out_top_id   = top_id;
    if (out_top_prob) *out_top_prob = probs[top_id] * inv_sum;
    return entropy;
}

/* ============================================================================
 * 4. Offloaded Discrete Engram Hash Table & SIMD Bias Injection (CFM-06)
 * ============================================================================ */

/*
 * Seqlock protocol for CFEngramSlot:
 * - seq == 0: slot is uncommitted / empty
 * - seq is odd (1, 3, 5, ...): writer is actively copying payload into vectors[idx]
 * - seq is even >= 2 (2, 4, 6, ...): slot payload is committed and stable
 * Readers snapshot `s1 = seq.load(acquire)`, wait if `s1 == 0 || (s1 & 1)`,
 * copy `vectors[idx]` into a local stack/heap buffer, issue `atomic_thread_fence(acquire)`,
 * and verify `s2 == s1` before applying `simd_axpy_f32` to `out`.
 */
struct CFEngramSlot {
    std::atomic<uint64_t> key_hash{0};   /* 0 = empty slot */
    std::atomic<uint64_t> seq{0};        /* Seqlock version counter */
};

struct CFEngramTable {
    uint32_t              capacity;
    uint32_t              max_entries;
    uint32_t              mask;
    uint32_t              d_model;
    std::atomic<uint32_t> count{0};
    CFEngramSlot*         slots;
    float*                vectors; /* capacity * d_model contiguous float32 */
};

extern "C" CF_API uint64_t cf_fnv1a_hash64(const char* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    if (!data) return hash;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
        hash *= 1099511628211ULL;
    }
    return (hash == 0ULL) ? 1ULL : hash;
}

extern "C" CF_API CFEngramTable* cf_engram_create(uint32_t capacity_pow2, uint32_t d_model) {
    if (d_model == 0 || capacity_pow2 == 0) return nullptr;
    uint32_t cap = next_pow2(capacity_pow2 < 4 ? 4 : capacity_pow2);
    auto* tbl = new (std::nothrow) CFEngramTable();
    if (!tbl) return nullptr;
    tbl->capacity = cap;
    tbl->max_entries = capacity_pow2;
    tbl->mask = cap - 1;
    tbl->d_model = d_model;
    tbl->slots = new (std::nothrow) CFEngramSlot[cap];
    if (!tbl->slots) {
        delete tbl;
        return nullptr;
    }
    tbl->vectors = static_cast<float*>(std::calloc(static_cast<size_t>(cap) * d_model, sizeof(float)));
    if (!tbl->vectors) {
        delete[] tbl->slots;
        delete tbl;
        return nullptr;
    }
    return tbl;
}

extern "C" CF_API void cf_engram_destroy(CFEngramTable* tbl) {
    if (!tbl) return;
    std::free(tbl->vectors);
    delete[] tbl->slots;
    delete tbl;
}

extern "C" CF_API int cf_engram_Upsert(CFEngramTable* tbl, uint64_t key_hash, const float* bias_vec) {
    if (!tbl || !bias_vec) return 0;
    if (key_hash == 0ULL) key_hash = 1ULL;

    uint32_t idx = static_cast<uint32_t>(key_hash) & tbl->mask;
    for (uint32_t probe = 0; probe < tbl->capacity; ++probe) {
        uint64_t expected_key = 0ULL;
        uint64_t cur_key = tbl->slots[idx].key_hash.load(std::memory_order_acquire);

        if (cur_key == 0ULL) {
            if (tbl->count.load(std::memory_order_acquire) >= tbl->max_entries) {
                return 0; /* Reached requested logical capacity */
            }
            if (tbl->slots[idx].key_hash.compare_exchange_strong(
                    expected_key, key_hash, std::memory_order_acq_rel)) {
                /* Claimed empty slot: seq is 0, set to 1 (odd = writing), copy payload, then set to 2 (even = committed) */
                tbl->slots[idx].seq.store(1ULL, std::memory_order_release);
                std::atomic_thread_fence(std::memory_order_release);
                std::memcpy(
                    tbl->vectors + static_cast<size_t>(idx) * tbl->d_model,
                    bias_vec,
                    tbl->d_model * sizeof(float)
                );
                tbl->slots[idx].seq.store(2ULL, std::memory_order_release);
                tbl->count.fetch_add(1, std::memory_order_relaxed);
                return 1;
            }
            cur_key = expected_key;
        }

        if (cur_key == key_hash) {
            /* Spin-acquire exclusive write lock by advancing even seq >= 2 to odd seq + 1 */
            uint64_t s = tbl->slots[idx].seq.load(std::memory_order_acquire);
            for (;;) {
                if (s >= 2ULL && (s & 1ULL) == 0ULL) {
                    if (tbl->slots[idx].seq.compare_exchange_weak(
                            s, s + 1ULL, std::memory_order_acq_rel, std::memory_order_acquire)) {
                        break;
                    }
                } else {
                    std::this_thread::yield();
                    s = tbl->slots[idx].seq.load(std::memory_order_acquire);
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
            std::memcpy(
                tbl->vectors + static_cast<size_t>(idx) * tbl->d_model,
                bias_vec,
                tbl->d_model * sizeof(float)
            );
            tbl->slots[idx].seq.store(s + 2ULL, std::memory_order_release);
            return 1;
        }
        idx = (idx + 1) & tbl->mask;
    }
    return 0; /* Table full */
}

extern "C" CF_API int cf_engram_lookup_and_inject_f32(
    const CFEngramTable* tbl,
    uint64_t             key_hash,
    const float*         x,
    const float*         gamma,
    float*               out,
    float                scale,
    float                eps
) {
    if (!tbl || !x || !gamma || !out) return 0;
    if (key_hash == 0ULL) key_hash = 1ULL;

    const uint32_t d = tbl->d_model;
    float sum_sq = simd_dot_f32(x, x, d);
    float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(d)) + eps);

    for (uint32_t j = 0; j < d; ++j) {
        out[j] = (x[j] * inv_rms) * gamma[j];
    }

    uint32_t idx = static_cast<uint32_t>(key_hash) & tbl->mask;
    for (uint32_t probe = 0; probe < tbl->capacity; ++probe) {
        uint64_t cur = tbl->slots[idx].key_hash.load(std::memory_order_acquire);
        if (cur == 0ULL) {
            return 0; /* Miss */
        }
        if (cur == key_hash) {
            /* Optimistic Seqlock read: retry until an even sequence number >= 2 remains unchanged across memcpy */
            const float* bias_ptr = tbl->vectors + static_cast<size_t>(idx) * d;
            float stack_bias[256];
            std::vector<float> heap_bias;
            float* local_ptr = stack_bias;
            if (d > 256) {
                heap_bias.resize(d);
                local_ptr = heap_bias.data();
            }
            for (;;) {
                uint64_t s1 = tbl->slots[idx].seq.load(std::memory_order_acquire);
                if (s1 < 2ULL || (s1 & 1ULL) != 0ULL) {
                    std::this_thread::yield();
                    continue;
                }
                std::memcpy(local_ptr, bias_ptr, d * sizeof(float));
                std::atomic_thread_fence(std::memory_order_acquire);
                uint64_t s2 = tbl->slots[idx].seq.load(std::memory_order_acquire);
                if (s1 == s2) {
                    break;
                }
            }
            simd_axpy_f32(out, local_ptr, scale, d);
            return 1; /* Hit & Injected */
        }
        idx = (idx + 1) & tbl->mask;
    }
    return 0;
}

extern "C" CF_API uint32_t cf_engram_count(const CFEngramTable* tbl) {
    return tbl ? tbl->count.load(std::memory_order_acquire) : 0;
}

