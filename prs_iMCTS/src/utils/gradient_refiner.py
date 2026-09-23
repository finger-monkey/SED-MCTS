from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import copy

import numpy as np


@dataclass
class RefinementEvent:
    prefix: List[str]
    action: str
    reward: float
    improved: bool
    delta: float
    metadata: Optional[Dict] = None


class TwoStageRefiner:










    def __init__(
        self,
        optimizer,
        stat_policy,
        arity_dict: Dict[str, int],
        delta_threshold: float = 1e-4,
        fast_const_steps: int = 5,
        topk_subtrees: int = 3,
        topm_actions: int = 3,
        pde_mode: str = "none",
        advection_beta: float = 1.0,
        physics_lambda: float = 0.0,
        stat_feedback_weight: float = 0.22,
        stat_feedback_max_events: int = 8,
        ablation_feedback_weight: float = 0.15,
        ablation_feedback_max_edges: int = 8,
    ) -> None:
        self.optimizer = optimizer
        self.stat_policy = stat_policy
        self.arity_dict = arity_dict
        self.delta_threshold = float(delta_threshold)
        self.fast_const_steps = int(fast_const_steps)
        self.topk_subtrees = int(topk_subtrees)
        self.topm_actions = int(topm_actions)
        self.pde_mode = str(pde_mode).lower().strip()
        self.advection_beta = float(advection_beta)
        self.physics_lambda = max(0.0, float(physics_lambda))
        self.stat_feedback_weight = max(0.0, float(stat_feedback_weight))
        self.stat_feedback_max_events = max(0, int(stat_feedback_max_events))
        self.ablation_feedback_weight = max(0.0, float(ablation_feedback_weight))
        self.ablation_feedback_max_edges = max(0, int(ablation_feedback_max_edges))

    def tune_constants(self, exp_tree_obj) -> Tuple[str, float]:
        expr_star, reward_star = self.optimizer.optimize_constants(exp_tree_obj)
        return expr_star, float(reward_star)

    def score_subtrees(self, exp_tree_obj) -> List[Tuple[int, float]]:
        op_list = list(getattr(exp_tree_obj, "op_list", []))
        op_count = len(op_list)
        if op_count == 0:
            return []

        base_expr, base_reward = self.optimizer.optimize_constants(exp_tree_obj)
        if not np.isfinite(base_reward):
            base_reward = 0.0

        base_pde_loss = self._pde_loss(base_expr)

        scores: List[Tuple[int, float]] = []
        for idx, op in enumerate(op_list):
            if self.arity_dict.get(op, 0) == 0:
                continue

            try:
                delta_reward = self._finite_diff_subtree_reward(exp_tree_obj, idx, base_reward)
                delta_pde = self._finite_diff_subtree_pde(exp_tree_obj, idx, base_pde_loss)
            except Exception:
                delta_reward, delta_pde = 0.0, 0.0

            score = float(delta_reward + self.physics_lambda * delta_pde)
            scores.append((idx, score))

        if not scores:
            scores = [(idx, 0.0) for idx in range(min(op_count, self.topk_subtrees))]

        scores.sort(key=lambda x: x[1], reverse=True)

        _cap = max(64, self.topk_subtrees, self.ablation_feedback_max_edges)
        return scores[: min(len(scores), _cap)]

    def _subtree_size(self, path: List[str], start: int) -> int:
        if start >= len(path):
            return 0

        def _consume(i: int) -> int:
            if i >= len(path):
                return i
            op = path[i]
            ar = self.arity_dict.get(op, 0)
            cur = i + 1
            for _ in range(ar):
                cur = _consume(cur)
            return cur

        end = _consume(start)
        return max(0, end - start)

    def _safe_eval_vec(self, expr_str: str):
        try:
            f = eval(compile(f"lambda x: {expr_str}", "<expr-refine>", "eval"), self.optimizer.context)
            y = f(self.optimizer.x_train)
            y = np.asarray(y)
            if y.ndim == 0:
                y = np.full_like(self.optimizer.y_train, float(y), dtype=float)
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            if y.shape != self.optimizer.y_train.shape:
                y = np.reshape(y, self.optimizer.y_train.shape)
            return y
        except Exception:
            return None

    def _infer_grid_shape(self) -> Optional[Tuple[int, int]]:
        x_train = np.asarray(self.optimizer.x_train)
        if x_train.ndim != 2 or x_train.shape[0] < 2:
            return None

        x_vals = np.asarray(x_train[0]).reshape(-1)
        t_vals = np.asarray(x_train[1]).reshape(-1)
        nx = np.unique(x_vals).shape[0]
        nt = np.unique(t_vals).shape[0]
        if nx * nt != x_vals.shape[0]:
            return None
        return nt, nx

    def _advection_residual_loss(self, y_pred_flat: np.ndarray) -> float:
        grid = self._infer_grid_shape()
        if grid is None:
            return 0.0
        nt, nx = grid

        x_train = np.asarray(self.optimizer.x_train)
        x_vals = np.asarray(x_train[0]).reshape(-1)
        t_vals = np.asarray(x_train[1]).reshape(-1)

        x_unique = np.unique(x_vals)
        t_unique = np.unique(t_vals)
        if len(x_unique) < 3 or len(t_unique) < 3:
            return 0.0

        dx = float(np.mean(np.diff(x_unique)))
        dt = float(np.mean(np.diff(t_unique)))
        if dx == 0.0 or dt == 0.0:
            return 0.0

        u = np.reshape(y_pred_flat, (nt, nx))


        u_t = (u[2:, :] - u[:-2, :]) / (2.0 * dt)
        u_x = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)

        u_t_c = u_t[:, 1:-1]
        u_x_c = u_x[1:-1, :]

        beta = self.advection_beta
        res = u_t_c + beta * u_x_c
        return float(np.mean(res * res))

    def _pde_loss(self, expr_str: str) -> float:
        if self.pde_mode != "advection" or self.physics_lambda <= 0.0:
            return 0.0
        y = self._safe_eval_vec(expr_str)
        if y is None:
            return 0.0
        return self._advection_residual_loss(y)

    def _finite_diff_subtree_reward(self, exp_tree_obj, idx: int, base_reward: float) -> float:
        op_list = list(getattr(exp_tree_obj, "op_list", []))
        if not (0 <= idx < len(op_list)):
            return 0.0

        op = op_list[idx]
        arity = self.arity_dict.get(op, 0)

        eps = 1e-3
        subtree_sz = self._subtree_size(op_list, idx)

        if arity == 2 and subtree_sz > 1:
            left_start = idx + 1
            left_size = self._subtree_size(op_list, left_start)
            right_start = left_start + left_size

            if right_start < len(op_list):
                mutant = copy.deepcopy(exp_tree_obj)
                m = list(mutant.op_list)
                m[idx] = '+'
                mutant.op_list = m
                _, r_plus = self.optimizer.optimize_constants(mutant)

                mutant2 = copy.deepcopy(exp_tree_obj)
                m2 = list(mutant2.op_list)
                m2[idx] = '-'
                mutant2.op_list = m2
                _, r_minus = self.optimizer.optimize_constants(mutant2)

                if np.isfinite(r_plus) and np.isfinite(r_minus):
                    return abs((float(r_plus) - float(r_minus)) / (2.0 * eps))

        mutant = copy.deepcopy(exp_tree_obj)
        m = list(mutant.op_list)
        replace_token = 'R' if 'R' in m else None
        if replace_token is None:
            replace_token = 'x0' if 'x0' in m else m[idx]
        m[idx:idx + max(1, subtree_sz)] = [replace_token]
        mutant.op_list = m

        _, r_ablate = self.optimizer.optimize_constants(mutant)
        if not np.isfinite(r_ablate):
            return 0.0
        return max(0.0, float(base_reward) - float(r_ablate))

    def _finite_diff_subtree_pde(self, exp_tree_obj, idx: int, base_pde_loss: float) -> float:
        if self.pde_mode != "advection" or self.physics_lambda <= 0.0:
            return 0.0

        op_list = list(getattr(exp_tree_obj, "op_list", []))
        if not (0 <= idx < len(op_list)):
            return 0.0

        subtree_sz = self._subtree_size(op_list, idx)
        mutant = copy.deepcopy(exp_tree_obj)
        m = list(mutant.op_list)
        replace_token = 'R' if 'R' in m else None
        if replace_token is None:
            replace_token = 'x0' if 'x0' in m else m[idx]
        m[idx:idx + max(1, subtree_sz)] = [replace_token]
        mutant.op_list = m

        expr_mut, _ = self.optimizer.optimize_constants(mutant)
        pde_loss_mut = self._pde_loss(expr_mut)
        return max(0.0, float(base_pde_loss) - float(pde_loss_mut))

    def _candidate_actions_same_arity(self, current_op: str, all_ops: Sequence[str]) -> List[str]:
        arity = self.arity_dict.get(current_op, None)
        if arity is None:
            return []
        cands = [op for op in all_ops if self.arity_dict.get(op, -1) == arity and op != current_op]
        return cands[: max(1, self.topm_actions)]

    def ablation_feedback_to_stat_policy(
        self,
        exp_tree_obj,
        ranked_scores: List[Tuple[int, float]],
        base_reward: float,
    ) -> int:




        if self.ablation_feedback_weight <= 0.0 or not ranked_scores:
            return 0
        op_list = list(getattr(exp_tree_obj, "op_list", []))
        raw = [float(s) for _, s in ranked_scores[: self.ablation_feedback_max_edges]]
        if not raw:
            return 0
        max_s = max(raw)
        if max_s <= 1e-12:
            return 0
        count = 0
        br = float(base_reward) if np.isfinite(float(base_reward)) else 0.0
        for idx, score in ranked_scores[: self.ablation_feedback_max_edges]:
            if not (0 <= idx < len(op_list)):
                continue
            op = op_list[idx]
            if self.arity_dict.get(op, 0) == 0:
                continue
            norm = float(score) / max_s

            w = self.ablation_feedback_weight * (norm ** 0.5)
            if w <= 1e-12:
                continue
            prefix = list(op_list[:idx])
            self.stat_policy.record_edge_feedback(
                prefix=prefix,
                action=op,
                reward=br,
                weight=w,
                force_fail=False,
            )
            count += 1
        return count

    def structural_refine(self, exp_tree_obj, all_ops: Sequence[str]) -> Tuple[Optional[str], float, List[RefinementEvent], int]:
        base_expr, base_reward = self.optimizer.optimize_constants(exp_tree_obj)
        best_expr, best_reward = base_expr, float(base_reward)
        events: List[RefinementEvent] = []

        ranked_full = self.score_subtrees(exp_tree_obj)
        ablation_written = self.ablation_feedback_to_stat_policy(exp_tree_obj, ranked_full, float(base_reward))
        ranked_nodes = ranked_full[: max(1, self.topk_subtrees)]

        for node_idx, _score in ranked_nodes:
            op_list = list(getattr(exp_tree_obj, "op_list", []))
            if not (0 <= node_idx < len(op_list)):
                continue

            old_op = op_list[node_idx]
            for new_op in self._candidate_actions_same_arity(old_op, all_ops):
                mutant = copy.deepcopy(exp_tree_obj)
                mutant.op_list[node_idx] = new_op

                try:
                    expr_new, reward_new = self.optimizer.optimize_constants(mutant)
                    reward_new = float(reward_new)
                except Exception:
                    continue

                delta = reward_new - best_reward
                improved = delta > self.delta_threshold


                prefix_path = list(op_list[:node_idx])
                events.append(
                    RefinementEvent(
                        prefix=prefix_path,
                        action=new_op,
                        reward=reward_new,
                        improved=improved,
                        delta=delta,
                        metadata={"node_idx": node_idx, "old_op": old_op, "new_op": new_op},
                    )
                )

                if improved:
                    best_expr, best_reward = expr_new, reward_new

        return best_expr, best_reward, events, ablation_written

    def feedback_to_stat_policy(self, events: List[RefinementEvent]) -> int:

        improved = [ev for ev in events if ev.improved]
        if not improved or self.stat_feedback_weight <= 0.0:
            return 0
        improved.sort(key=lambda e: e.delta, reverse=True)
        cap = self.stat_feedback_max_events
        if cap > 0:
            improved = improved[:cap]
        w = self.stat_feedback_weight
        count = 0
        for ev in improved:
            self.stat_policy.record_edge_feedback(
                prefix=ev.prefix,
                action=ev.action,
                reward=ev.reward,
                weight=w,
                force_fail=False,
            )
            count += 1
        return count
