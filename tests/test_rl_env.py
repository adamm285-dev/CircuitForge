"""Unit tests for the public `circuitforge.rl_env` verifiable RL environment."""

from __future__ import annotations

from circuitforge.rl_env import MultiTurnTaskSpec, VerifiableCodingEnv


def test_verifiable_coding_env_lifecycle() -> None:
    """Verify reset, step, and custom verifier reward calculation."""

    def dummy_verifier(history: list[str]) -> tuple[float, bool]:
        joined = "\n".join(history)
        if "FORBIDDEN" in joined:
            return 0.5, False
        if "def solve():" in joined and "SUBMIT" in joined:
            return 1.0, True
        if "def solve():" in joined:
            return 0.5, True
        return 0.0, True

    spec = MultiTurnTaskSpec(
        task_id="toy_01",
        prompt="Implement solve() and call SUBMIT.",
        max_turns=5,
        verifier_fn=dummy_verifier,
    )
    env = VerifiableCodingEnv(spec)
    obs, info = env.reset(seed=1)
    assert "Implement solve()" in obs
    assert info["turn"] == 0

    obs1, r1, term1, trunc1, info1 = env.step("def solve(): return 42")
    assert not term1 and not trunc1
    assert r1 > 0.4

    obs2, r2, term2, trunc2, info2 = env.step("SUBMIT")
    assert term2 and not trunc2
    assert r2 > 0.9
    assert info2["reward_breakdown"].tests_passed_ratio == 1.0
    assert info2["reward_breakdown"].invariants_preserved is True


def test_verifiable_coding_env_invariant_violation_zeroes_pass_reward() -> None:
    """Verify breaking an invariant zeroes out the test pass reward."""
    spec = MultiTurnTaskSpec(
        task_id="toy_02",
        prompt="Test invariant check.",
        max_turns=3,
        verifier_fn=lambda hist: (1.0, False),
    )
    env = VerifiableCodingEnv(spec)
    env.reset()
    _, r, term, _, info = env.step("bad edit")
    assert not info["reward_breakdown"].invariants_preserved
    assert r < 0.1
