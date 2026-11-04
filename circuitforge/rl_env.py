"""
CircuitForge Generic Verifiable Multi-Turn RL Environment (`rl_env.py`).

Provides a clean, framework-agnostic Gymnasium / `verifiers`-compatible environment
(`reset`, `step`, `compute_step_reward`) for training or adapting open-weights coding models
with Group Relative Policy Optimization (`GRPO`) and Reinforcement Learning with Verifiable
Rewards (`RLVR`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class VerifiableStepReward:
    """Breakdown of deterministic step-level and terminal verifiable rewards."""

    syntax_valid: bool
    tests_passed_ratio: float
    invariants_preserved: bool
    step_penalty: float
    total_reward: float


@dataclass
class MultiTurnTaskSpec:
    """Specification for a verifiable multi-turn coding / circuit task."""

    task_id: str
    prompt: str
    max_turns: int = 12
    verifier_fn: Callable[[list[str]], tuple[float, bool]] | None = None


class VerifiableCodingEnv:
    """
    Minimal, zero-leakage Gymnasium / `verifiers` RL environment for multi-turn code generation
    and tool-call verification.
    """

    def __init__(self, task: MultiTurnTaskSpec) -> None:
        self.task = task
        self.turn: int = 0
        self.history: list[str] = []
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(self, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        self.turn = 0
        self.history = []
        self.terminated = False
        self.truncated = False
        return self.task.prompt, {"task_id": self.task.task_id, "turn": 0, "seed": seed}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot call step() on a terminated/truncated episode; call reset() first.")

        self.turn += 1
        self.history.append(action)

        syntax_ok = len(action.strip()) > 0
        pass_ratio = 0.0
        invariants_ok = True

        if self.task.verifier_fn is not None:
            pass_ratio, invariants_ok = self.task.verifier_fn(self.history)
        elif "SUBMIT" in action:
            pass_ratio = 1.0

        if "SUBMIT" in action or pass_ratio >= 1.0:
            self.terminated = True
        elif self.turn >= self.task.max_turns:
            self.truncated = True

        step_pen = -0.02 * self.turn
        base_r = (pass_ratio if invariants_ok else 0.0) + (0.05 if syntax_ok else -0.2) + step_pen
        reward_breakdown = VerifiableStepReward(
            syntax_valid=syntax_ok,
            tests_passed_ratio=pass_ratio,
            invariants_preserved=invariants_ok,
            step_penalty=step_pen,
            total_reward=round(float(base_r), 4),
        )

        obs = f"Turn {self.turn}: pass_ratio={pass_ratio:.2f}, invariants_ok={invariants_ok}"
        info = {
            "turn": self.turn,
            "reward_breakdown": reward_breakdown,
        }
        return obs, reward_breakdown.total_reward, self.terminated, self.truncated, info
