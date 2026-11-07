/*
 * CircuitForge Native C++ Zero-Copy Tensor Engine & Telemetry Ring Buffer
 * File: src/engine/circuitforge_engine.h
 *
 * Implements:
 * 1. Cross-platform Zero-Copy Memory-Mapped Weight Access (`MapViewOfFile` / `mmap`)
 * 2. High-Velocity AVX2/FMA SIMD Kernels for In-Place Rank-1 Null-Space Updates,
 *    Directional Feature Ablation, RMSNorm, and Matrix-Vector Projection.
 * 3. Zero-Alloc Lock-Free SPSC Telemetry Ring Buffer emitting layer activation
 *    norms, router entropy metrics, and expert selection distributions.
 */

#ifndef CIRCUITFORGE_ENGINE_H
#define CIRCUITFORGE_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #ifdef CF_BUILD_DLL
    #define CF_API __declspec(dllexport)
  #else
    #define CF_API __declspec(dllimport)
  #endif
#else
  #define CF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Telemetry Event Struct & Lock-Free Ring Buffer API
 * ============================================================================ */

typedef struct {
    uint64_t timestamp_ns;
    uint32_t step_id;
    uint32_t layer_idx;
    float    activation_l2_norm;
    float    router_entropy;
    float    steering_alpha;
    uint32_t top_expert_id;
    float    top_expert_weight;
    uint32_t anomaly_flag;
} CFTelemetryEvent;

typedef struct CFTelemetryRingBuffer CFTelemetryRingBuffer;

CF_API CFTelemetryRingBuffer* cf_ringbuf_create(uint32_t capacity_pow2);
CF_API void                   cf_ringbuf_destroy(CFTelemetryRingBuffer* rb);
CF_API int                    cf_ringbuf_push(CFTelemetryRingBuffer* rb, const CFTelemetryEvent* ev);
CF_API int                    cf_ringbuf_pop(CFTelemetryRingBuffer* rb, CFTelemetryEvent* out_ev);
CF_API uint32_t               cf_ringbuf_size(const CFTelemetryRingBuffer* rb);
CF_API uint32_t               cf_ringbuf_capacity(const CFTelemetryRingBuffer* rb);

/* ============================================================================
 * 2. Zero-Copy Memory-Mapped Checkpoint Handle
 * ============================================================================ */

typedef struct CFMmapFile CFMmapFile;

CF_API CFMmapFile* cf_mmap_open(const char* filepath, int writable);
CF_API void*       cf_mmap_data(CFMmapFile* mf);
CF_API size_t      cf_mmap_size(const CFMmapFile* mf);
CF_API int         cf_mmap_flush(CFMmapFile* mf);
CF_API void        cf_mmap_close(CFMmapFile* mf);

/* ============================================================================
 * 3. High-Velocity SIMD Tensor & Weight Surgery Kernels (F32)
 * ============================================================================ */

/*
 * In-place Rank-1 weight update:
 *   W[i, j] += scale * u[i] * v[j]
 * where W is (d_out, d_in) row-major float32, u is (d_out,), v is (d_in,).
 * Returns Frobenius norm of the applied delta_W.
 */
CF_API float cf_rank1_update_f32(
    float*       W,
    uint32_t     d_out,
    uint32_t     d_in,
    const float* u_vec,
    const float* v_vec,
    float        scale
);

/*
 * Null-Space Constrained Rank-1 Weight Surgery in-place on W (d_out, d_in):
 *   k_proj = k_star - sum_{b=0..r-1} (basis[b] . k_star) * basis[b]
 *   W += alpha * (v_unit (x) k_proj) / ||k_proj||^2
 * where `basis` is an orthonormal row-major matrix (r, d_in) spanning preserved keys K_0.
 * Populates `out_align_before`, `out_align_after`, and returns Frobenius delta norm.
 */
CF_API float cf_nullspace_rank1_surgery_f32(
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
);

/*
 * In-place Directional Feature Ablation:
 *   W <- (I - strength * r_unit * r_unit^T) * W
 * where W is (d_out, d_in) and r_unit is normalized from `target_dir` (d_out,).
 * Returns Frobenius norm of the removed component.
 */
CF_API float cf_directional_ablation_f32(
    float*       W,
    uint32_t     d_out,
    uint32_t     d_in,
    const float* target_dir,
    float        strength
);

/*
 * Fast SIMD RMSNorm + Matrix-Vector Projection:
 *   x_norm[j] = (x[j] / sqrt(mean(x^2) + eps)) * gamma[j]
 *   out[i]    = dot(W[i, :], x_norm[:])
 */
CF_API void cf_rmsnorm_matvec_f32(
    const float* W,
    const float* x,
    const float* gamma,
    float*       out,
    uint32_t     d_out,
    uint32_t     d_in,
    float        eps
);

/*
 * Computes Shannon entropy of router logits (with numerically stable softmax)
 * and identifies top-1 expert index and probability.
 */
CF_API float cf_router_entropy_f32(
    const float* logits,
    uint32_t     num_experts,
    uint32_t*    out_top_id,
    float*       out_top_prob
);

/* ============================================================================
 * 4. Offloaded Discrete Engram Hash Table & SIMD Bias Injection (CFM-06)
 * ============================================================================ */

typedef struct CFEngramTable CFEngramTable;

CF_API CFEngramTable* cf_engram_create(uint32_t capacity_pow2, uint32_t d_model);
CF_API void           cf_engram_destroy(CFEngramTable* tbl);
CF_API uint64_t       cf_fnv1a_hash64(const char* data, size_t len);
CF_API int            cf_engram_Upsert(CFEngramTable* tbl, uint64_t key_hash, const float* bias_vec);
CF_API int            cf_engram_lookup_and_inject_f32(
    const CFEngramTable* tbl,
    uint64_t             key_hash,
    const float*         x,
    const float*         gamma,
    float*               out,
    float                scale,
    float                eps
);
CF_API uint32_t       cf_engram_count(const CFEngramTable* tbl);

#ifdef __cplusplus
}
#endif

#endif /* CIRCUITFORGE_ENGINE_H */
