from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
import math


Prefix = Tuple[str, ...]


class StatisticalPolicyValue:


    def __init__(
        self,
        alpha: float = 1.0,
        tau: float = 1.0,
        beta: float = 2.0,
        failure_penalty: float = 2.5,
        failure_reward_threshold: float = 0.05,
        min_samples_for_bias: int = 1,
    ) -> None:
        self.alpha = max(alpha, 1e-8)
        self.tau = max(tau, 1e-8)
        self.beta = beta
        self.failure_penalty = max(failure_penalty, 0.0)
        self.failure_reward_threshold = failure_reward_threshold
        self.min_samples_for_bias = max(min_samples_for_bias, 1)

        self.sa_count: Dict[Prefix, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.sa_reward_sum: Dict[Prefix, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.sa_fail_count: Dict[Prefix, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self.s_count: Dict[Prefix, float] = defaultdict(float)
        self.s_value_sum: Dict[Prefix, float] = defaultdict(float)

    def _is_failure(self, reward: float) -> bool:
        return (not math.isfinite(reward)) or (reward <= self.failure_reward_threshold)

    def record_episode(self, path: Sequence[str], reward: float) -> None:
        r = float(reward) if math.isfinite(float(reward)) else -1.0
        failed = self._is_failure(r)

        prefix: List[str] = []
        for action in path:
            state = tuple(prefix)
            self.sa_count[state][action] += 1
            self.sa_reward_sum[state][action] += r
            if failed:
                self.sa_fail_count[state][action] += 1

            self.s_count[state] += 1
            self.s_value_sum[state] += r
            prefix.append(action)

        terminal_state = tuple(prefix)
        self.s_count[terminal_state] += 1
        self.s_value_sum[terminal_state] += r

    def record_weighted_episode(self, path: Sequence[str], reward: float, weight: float = 1.0, force_fail: bool = False) -> None:

        w = max(float(weight), 0.0)
        if w <= 0.0:
            return

        r = float(reward) if math.isfinite(float(reward)) else -1.0
        failed = force_fail or self._is_failure(r)

        prefix: List[str] = []
        for action in path:
            state = tuple(prefix)
            self.sa_count[state][action] += w
            self.sa_reward_sum[state][action] += w * r
            if failed:
                self.sa_fail_count[state][action] += w

            self.s_count[state] += w
            self.s_value_sum[state] += w * r
            prefix.append(action)

        terminal_state = tuple(prefix)
        self.s_count[terminal_state] += w
        self.s_value_sum[terminal_state] += w * r

    def record_edge_feedback(self, prefix: Sequence[str], action: str, reward: float, weight: float = 1.0, force_fail: bool = False) -> None:

        w = max(float(weight), 0.0)
        if w <= 0.0:
            return

        r = float(reward) if math.isfinite(float(reward)) else -1.0
        failed = force_fail or self._is_failure(r)

        state = tuple(prefix)
        self.sa_count[state][action] += w
        self.sa_reward_sum[state][action] += w * r
        if failed:
            self.sa_fail_count[state][action] += w


        self.s_count[state] += w
        self.s_value_sum[state] += w * r

    def get_state_value(self, prefix: Sequence[str]) -> float:
        state = tuple(prefix)
        n = self.s_count.get(state, 0)
        if n <= 0:
            return 0.0
        return self.s_value_sum[state] / n

    def get_action_stats(self, prefix: Sequence[str], action: str) -> Tuple[float, float, float]:
        state = tuple(prefix)
        n_sa = self.sa_count[state].get(action, 0)
        if n_sa <= 0:
            return 0, 0.0, 0.0
        avg_reward = self.sa_reward_sum[state][action] / n_sa
        fail_rate = self.sa_fail_count[state].get(action, 0) / n_sa
        return n_sa, avg_reward, fail_rate

    def get_action_risk(self, prefix: Sequence[str], action: str) -> float:
        _, _, fail_rate = self.get_action_stats(prefix, action)
        return fail_rate

    def get_action_prior(self, prefix: Sequence[str], actions: Iterable[str]) -> Dict[str, float]:
        state = tuple(prefix)
        actions = list(actions)
        if not actions:
            return {}

        weights: Dict[str, float] = {}
        for action in actions:
            n_sa = self.sa_count[state].get(action, 0)

            if n_sa < self.min_samples_for_bias:
                weights[action] = 1.0
                continue

            avg_reward = self.sa_reward_sum[state][action] / max(n_sa, 1)
            fail_rate = self.sa_fail_count[state].get(action, 0) / max(n_sa, 1)





            success_evidence = max(0.0, n_sa * (1.0 - fail_rate))
            count_term = (success_evidence + self.alpha) ** self.tau
            score_term = math.exp(self.beta * avg_reward - self.failure_penalty * fail_rate)
            weights[action] = max(count_term * score_term, 1e-12)

        z = sum(weights.values())
        if z <= 0:
            uni = 1.0 / len(actions)
            return {a: uni for a in actions}
        return {a: w / z for a, w in weights.items()}

    def debug_state_summary(self, prefix: Sequence[str], actions: Iterable[str], top_k: int = 5) -> List[str]:
        actions = list(actions)
        prior = self.get_action_prior(prefix, actions)
        rows = []
        for a in actions:
            n, avg_r, fail = self.get_action_stats(prefix, a)
            rows.append((a, prior.get(a, 0.0), n, avg_r, fail))
        rows.sort(key=lambda x: x[1], reverse=True)

        lines: List[str] = []
        for a, p, n, avg_r, fail in rows[: max(1, top_k)]:
            lines.append(f"{a}: p={p:.3f}, n={n}, avgR={avg_r:.3f}, fail={fail:.2f}")
        return lines
