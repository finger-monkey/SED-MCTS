from __future__ import annotations
import copy
import random
from typing import List, Optional, Tuple, Dict

import numpy as np

from prs_iMCTS.gp import GPManager
from prs_iMCTS.src import ExpTree, Exp_Queue, Optimizer, StatisticalPolicyValue


class MCTS_Node:
    def __init__(
        self,
        mcts: "MCTS",
        parent: Optional["MCTS_Node"] = None,
        move: str = "",
    ) -> None:
        self.mcts = mcts
        self.parent = parent
        self.move = move
        self.children: List["MCTS_Node"] = []
        self.visits: int = 0
        self.path_queue = Exp_Queue(max_size=mcts.K)
        self.unexpanded_moves: List[str] = []
        self.is_terminal: bool = False

    def get_prefix_path(self) -> List[str]:
        path: List[str] = []
        cur: Optional["MCTS_Node"] = self
        while cur is not None and cur.parent is not None:
            path.append(cur.move)
            cur = cur.parent
        path.reverse()
        return path

    def expand(self, state: ExpTree) -> Optional["MCTS_Node"]:
        if not self.unexpanded_moves:
            self.unexpanded_moves = list(state.available_ops)
        if not self.unexpanded_moves:
            return None

        prefix = self.get_prefix_path()
        prior = self.mcts.stat_policy.get_action_prior(prefix, self.unexpanded_moves)
        probs = np.array([prior.get(a, 0.0) for a in self.unexpanded_moves], dtype=float)

        if probs.sum() <= 1e-12:
            idx = random.randrange(len(self.unexpanded_moves))
        else:
            probs = probs / probs.sum()
            idx = int(np.random.choice(len(self.unexpanded_moves), p=probs))

        move = self.unexpanded_moves.pop(idx)
        child = MCTS_Node(mcts=self.mcts, parent=self, move=move)
        self.children.append(child)

        if self.mcts.verbose and self.mcts.debug_print_prior_every > 0 and self.mcts.count_num % self.mcts.debug_print_prior_every == 0:
            print(f"[PRIOR][expand] prefix={prefix} -> choose='{move}'")
            for line in self.mcts.stat_policy.debug_state_summary(prefix, [c.move for c in self.children] + self.unexpanded_moves, top_k=self.mcts.debug_top_k):
                print(f"  {line}")

        return child

    def q_value(self) -> float:
        if self.path_queue.list:
            return float(self.path_queue.best()[1])
        return float(self.mcts.stat_policy.get_state_value(self.get_prefix_path()))

    def puct(self) -> float:
        q = self.q_value()
        if self.parent is None:
            return q

        parent_visits = max(self.parent.visits, 1)
        visits = max(self.visits, 1)

        prefix = self.parent.get_prefix_path()
        sibling_actions = [c.move for c in self.parent.children]
        prior_dist = self.mcts.stat_policy.get_action_prior(prefix, sibling_actions)
        p_sa = prior_dist.get(self.move, 1.0 / max(len(sibling_actions), 1))
        risk = self.mcts.stat_policy.get_action_risk(prefix, self.move)

        u = self.mcts.c * p_sa * (np.sqrt(parent_visits) / (1.0 + visits))
        return q + u - self.mcts.risk_lambda * risk

    def choose(self) -> "MCTS_Node":
        valid = [c for c in self.children if not (c.is_terminal and c.visits > 0)]
        if not valid:
            valid = self.children
        return max(valid, key=lambda c: c.puct())

    def random_child(self) -> "MCTS_Node":
        valid = [c for c in self.children if not c.is_terminal]
        if valid:
            return random.choice(valid)
        return random.choice(self.children)

    def backpropagate(self, path: List[str], value: float) -> None:
        cur: Optional["MCTS_Node"] = self
        remain = list(path)
        while cur is not None:
            if not cur.path_queue.append(remain, value):
                break
            if cur.parent is not None:
                remain = [cur.move] + remain
            cur = cur.parent

    def propagate(self, path: List[str], value: float) -> None:
        cur: Optional["MCTS_Node"] = self
        for i, action in enumerate(path):
            cur = next((child for child in cur.children if child.move == action), None) if cur else None
            if cur is None:
                break
            cur.path_queue.append(path[i + 1 :], value)

    def is_leaf(self) -> bool:
        return not self.children or bool(self.unexpanded_moves)


class MCTS:
    def __init__(
        self,
        optimizer: Optimizer,
        gp_manager: GPManager,
        gp_rate: float = 0.2,
        mutation_rate: float = 0.2,
        exploration_rate: float = 0.2,
        K: int = 500,
        c: float = 4,
        gamma: float = 0.5,
        verbose: bool = False,
        succ_error_tol: float = 1e-6,
        stat_policy: Optional[StatisticalPolicyValue] = None,
        risk_lambda: float = 0.5,
        rollout_guided: bool = True,
        rollout_epsilon: float = 0.15,
        rollout_temperature: float = 1.0,
        debug_print_rollout_every: int = 100,
        debug_print_prior_every: int = 200,
        debug_print_stats_every: int = 200,
        debug_top_k: int = 5,
        reverse_search_enabled: bool = False,
        reverse_search_subtree_samples: int = 4,
        reverse_search_every: int = 25,
        reverse_mode: str = "contrastive",
        contrastive_enable_op_replace: bool = True,
        contrastive_reward_weight: float = 0.9,
        contrastive_penalty_weight: float = 0.8,
        reverse_topk_edges: int = 0,
        reverse_topm_actions: int = 0,
        reverse_winner_only: bool = True,
        reverse_min_improvement: float = 1e-4,
        reverse_complexity_lambda: float = 0.0,
        reverse_where_metric: str = "harmfulness",
        reverse_prior_topk_states: int = 3,
        reverse_prior_topm_actions: int = 3,
        reverse_physics_weight: float = 0.7,
        reverse_data_weight: float = 0.3,
    ) -> None:
        self.optimizer = optimizer
        self.gp_manager = gp_manager
        self.gp_rate = gp_rate
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.K = K
        self.c = c
        self.gamma = gamma
        self.verbose = verbose
        self.succ_error_tol = succ_error_tol
        self.risk_lambda = max(0.0, risk_lambda)

        self.stat_policy = stat_policy or StatisticalPolicyValue()

        self.rollout_guided = rollout_guided
        self.rollout_epsilon = float(np.clip(rollout_epsilon, 0.0, 1.0))
        self.rollout_temperature = max(float(rollout_temperature), 1e-6)

        self.debug_print_rollout_every = max(0, debug_print_rollout_every)
        self.debug_print_prior_every = max(0, debug_print_prior_every)
        self.debug_print_stats_every = max(0, debug_print_stats_every)
        self.debug_top_k = max(1, debug_top_k)

        self.reverse_search_enabled = bool(reverse_search_enabled)
        self.reverse_search_subtree_samples = int(reverse_search_subtree_samples)
        self.reverse_search_every = max(1, int(reverse_search_every))
        self.reverse_mode = str(reverse_mode).lower().strip()
        if self.reverse_mode not in {"ablation", "contrastive", "where_only", "prior_anchor"}:
            self.reverse_mode = "prior_anchor"
        self.contrastive_enable_op_replace = bool(contrastive_enable_op_replace)
        self.contrastive_reward_weight = max(0.0, float(contrastive_reward_weight))
        self.contrastive_penalty_weight = max(0.0, float(contrastive_penalty_weight))
        self.reverse_topk_edges = max(0, int(reverse_topk_edges))
        self.reverse_topm_actions = max(0, int(reverse_topm_actions))
        self.reverse_winner_only = bool(reverse_winner_only)
        self.reverse_min_improvement = max(0.0, float(reverse_min_improvement))
        self.reverse_complexity_lambda = max(0.0, float(reverse_complexity_lambda))
        self.reverse_where_metric = str(reverse_where_metric).lower().strip()
        if self.reverse_where_metric not in {"harmfulness", "repairability"}:
            self.reverse_where_metric = "harmfulness"
        self.reverse_prior_topk_states = max(1, int(reverse_prior_topk_states))
        self.reverse_prior_topm_actions = max(1, int(reverse_prior_topm_actions))
        self.reverse_physics_weight = max(0.0, float(reverse_physics_weight))
        self.reverse_data_weight = max(0.0, float(reverse_data_weight))
        self.reverse_called_count: int = 0
        self.reverse_gate_pass_count: int = 0
        self.reverse_updates_written_count: int = 0
        self.last_reverse_search_info: Dict[str, float] = {}
        self.reverse_diagnostic_evals: int = 0
        self.mutation_trial_evals: int = 0

        self.root = MCTS_Node(mcts=self)
        self.exp_queue = Exp_Queue(max_size=10)
        self.count_num: int = 0
        self.best_reward: float = -np.inf
        self.total_nodes: int = 1

    def _maybe_print_stats_snapshot(self) -> None:
        if not self.verbose or self.debug_print_stats_every <= 0:
            return
        if self.count_num % self.debug_print_stats_every != 0:
            return

        root_actions = [c.move for c in self.root.children]
        if not root_actions:
            return

        print(f"[STAT][iter={self.count_num}] root prior/risk snapshot:")
        for line in self.stat_policy.debug_state_summary([], root_actions, top_k=self.debug_top_k):
            print(f"  {line}")

    def search(self, exp_tree: ExpTree):
        node = self.root
        self.count_num += 1
        node.visits += 1
        state = copy.deepcopy(exp_tree)

        while not node.is_leaf():
            if random.random() < self.gp_rate:
                gp_reward = self._perform_mutation(node, state) if random.random() < self.mutation_rate else self._perform_crossover(node, state)
                self.best_reward = max(self.best_reward, gp_reward)

            node = node.random_child() if random.random() < self.exploration_rate else node.choose()
            node.visits += 1
            state.add_op(node.move)

        if not state.is_terminal():
            child = node.expand(state)
            if child is not None:
                node = child
            self.total_nodes += 1
            node.visits += 1
            state.add_op(node.move)

        sim_reward, path = self.rollout_once(state)
        self.best_reward = max(self.best_reward, sim_reward)

        if not path:
            node.is_terminal = True
        node.backpropagate(path, sim_reward)

        episode = node.get_prefix_path() + list(path)


        if self.reverse_search_enabled and path and self.count_num % self.reverse_search_every == 0:
            self.reverse_called_count += 1
            try:
                _ = self._choose_reverse_search_index(
                    state,
                    full_episode=episode,
                    local_path=list(path),
                    base_reward=float(sim_reward) if np.isfinite(sim_reward) else -1.0,
                )
                if self.verbose and self.debug_print_stats_every > 0 and self.count_num % self.debug_print_stats_every == 0:
                    print(
                        f"[REVERSE][iter={self.count_num}] "
                        f"diag_evals={self.reverse_diagnostic_evals} "
                        f"sampled={int(self.last_reverse_search_info.get('sampled_subtrees', 0))} "
                        f"chosen_delta={self.last_reverse_search_info.get('chosen_delta', float('nan')):.6f} "
                        f"called={self.reverse_called_count} "
                        f"passed={self.reverse_gate_pass_count} "
                        f"written={self.reverse_updates_written_count} "
                        f"metric={self.reverse_where_metric}"
                    )
            except Exception:
                pass

        self.stat_policy.record_episode(episode, sim_reward)

        if self.verbose and self.debug_print_rollout_every > 0 and self.count_num % self.debug_print_rollout_every == 0:
            mode = "guided" if self.rollout_guided else "random"
            print(f"[ROLLOUT][{mode}][iter={self.count_num}] prefix={node.get_prefix_path()} sampled_path={path} reward={sim_reward:.6f}")

        self._update_terminal_status(node)

        if node.parent is not None:
            for parent_path, parent_reward in node.parent.path_queue.list:
                node.parent.propagate(parent_path, parent_reward)

        self._maybe_print_stats_snapshot()
        return self.best_reward

    def _update_terminal_status(self, node: MCTS_Node) -> None:
        cur = node
        while cur and cur.parent:
            parent = cur.parent
            if parent.children and all(ch.is_terminal for ch in parent.children) and not parent.unexpanded_moves:
                if not parent.is_terminal:
                    parent.is_terminal = True
                    cur = parent
                else:
                    break
            else:
                break

    def _enumerate_subtrees(self, path: List[str]) -> List[Tuple[int, int]]:
        subtrees: List[Tuple[int, int]] = []
        i = 0
        while i < len(path):
            size = self.gp_manager.cal_subtree_size_at_index(path, i)
            if size <= 0:
                break
            subtrees.append((i, size))
            i += 1
        return subtrees

    def _evaluate_candidate_reward(self, state: ExpTree, candidate_path: List[str]) -> float:
        try:
            cloned = copy.deepcopy(state)
            for op in candidate_path:
                cloned.add_op(op)
            _, reward = self.optimizer.optimize_constants(cloned)
            self.reverse_diagnostic_evals += 1
            return float(reward) if np.isfinite(reward) else -1.0
        except Exception:
            return -1.0

    def _path_complexity(self, path: List[str]) -> int:
        return len(path)

    def _effective_improvement(self, base_reward: float, candidate_reward: float, base_path: List[str], candidate_path: List[str]) -> float:
        raw = candidate_reward - base_reward
        dc = self._path_complexity(candidate_path) - self._path_complexity(base_path)
        return raw - self.reverse_complexity_lambda * max(0, dc)

    def _compute_physics_like_score(self, base_reward: float, candidate_reward: float, base_path: List[str], candidate_path: List[str]) -> float:

        data_term = candidate_reward - base_reward
        dc = self._path_complexity(candidate_path) - self._path_complexity(base_path)
        physics_term = -max(0, dc)
        return self.reverse_data_weight * data_term + self.reverse_physics_weight * physics_term

    def _apply_where_only_feedback(
        self,
        full_episode: List[str],
        local_path: List[str],
        local_idx: int,
        base_reward: float,
        candidate_reward: float,
        candidate_path: List[str],
    ) -> None:
        if not local_path:
            return


        prefix_len = max(0, len(full_episode) - len(local_path))
        edge_pos = prefix_len + max(0, min(local_idx, len(local_path) - 1))
        if edge_pos < 0 or edge_pos >= len(full_episode):
            return

        edge_prefix = full_episode[:edge_pos]
        old_action = full_episode[edge_pos]
        eff = self._effective_improvement(base_reward, candidate_reward, local_path, candidate_path)
        if eff > self.reverse_min_improvement:
            self.reverse_gate_pass_count += 1
            self.stat_policy.record_edge_feedback(edge_prefix, old_action, reward=base_reward, weight=self.contrastive_reward_weight, force_fail=False)
            self.reverse_updates_written_count += 1

    def _apply_contrastive_edge_feedback(
        self,
        full_episode: List[str],
        local_path: List[str],
        local_idx: int,
        candidate_action: str,
        base_reward: float,
        candidate_reward: float,
        candidate_path: List[str],
    ) -> None:
        if not local_path:
            return
        prefix_len = max(0, len(full_episode) - len(local_path))
        edge_pos = prefix_len + max(0, min(local_idx, len(local_path) - 1))
        if edge_pos < 0 or edge_pos >= len(full_episode):
            return

        edge_prefix = full_episode[:edge_pos]
        old_action = full_episode[edge_pos]

        rw = self.contrastive_reward_weight
        pw = self.contrastive_penalty_weight
        eff = self._effective_improvement(base_reward, candidate_reward, local_path, candidate_path)

        if eff > self.reverse_min_improvement:
            self.reverse_gate_pass_count += 1
            self.stat_policy.record_edge_feedback(edge_prefix, candidate_action, reward=candidate_reward, weight=rw, force_fail=False)
            self.reverse_updates_written_count += 1
            if (not self.reverse_winner_only) and pw > 0.0:
                self.stat_policy.record_edge_feedback(edge_prefix, old_action, reward=max(0.0, base_reward), weight=pw, force_fail=True)
                self.reverse_updates_written_count += 1
        elif eff < -self.reverse_min_improvement and (not self.reverse_winner_only):
            self.reverse_gate_pass_count += 1
            self.stat_policy.record_edge_feedback(edge_prefix, old_action, reward=base_reward, weight=rw, force_fail=False)
            self.reverse_updates_written_count += 1
            if pw > 0.0:
                self.stat_policy.record_edge_feedback(edge_prefix, candidate_action, reward=max(0.0, candidate_reward), weight=pw, force_fail=True)
                self.reverse_updates_written_count += 1

    def _apply_prior_anchored_reverse(self, state: ExpTree, full_episode: List[str], local_path: List[str], base_reward: float) -> Optional[int]:
        if not local_path:
            return None

        subtrees = self._enumerate_subtrees(local_path)
        if not subtrees:
            return None

        const_token = 'R' if 'R' in self.gp_manager.ops else ('x0' if 'x0' in self.gp_manager.ops else local_path[0])
        scored_states: List[Tuple[float, int, int]] = []
        for idx, size in subtrees:
            cf_const = local_path[:idx] + [const_token] + local_path[idx + size:]
            reward_const = self._evaluate_candidate_reward(state, cf_const)
            delta_const = base_reward - reward_const
            scored_states.append((delta_const, idx, size))

        scored_states.sort(key=lambda x: x[0], reverse=True)
        selected_states = scored_states[: min(self.reverse_prior_topk_states, len(scored_states))]

        best_idx = None
        best_gain = -np.inf
        for _delta, idx, _size in selected_states:
            edge_pos = max(0, len(full_episode) - len(local_path)) + idx
            edge_prefix = full_episode[:edge_pos]
            old_action = full_episode[edge_pos]

            root_op = local_path[idx]
            arity = self.gp_manager.arity_dict.get(root_op, -1)
            same_arity_ops = [op for op in self.gp_manager.ops if self.gp_manager.arity_dict.get(op, -2) == arity and op != old_action]
            if not same_arity_ops:
                continue

            prior = self.stat_policy.get_action_prior(edge_prefix, same_arity_ops)
            candidates = sorted(same_arity_ops, key=lambda a: prior.get(a, 0.0), reverse=True)[: self.reverse_prior_topm_actions]

            for cand in candidates:
                cf_path = list(local_path)
                cf_path[idx] = cand
                cf_reward = self._evaluate_candidate_reward(state, cf_path)
                gain = self._compute_physics_like_score(base_reward, cf_reward, local_path, cf_path)

                if gain > self.reverse_min_improvement:
                    self.reverse_gate_pass_count += 1
                    self.stat_policy.record_edge_feedback(edge_prefix, cand, reward=max(0.0, cf_reward), weight=self.contrastive_reward_weight, force_fail=False)
                    self.reverse_updates_written_count += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx

        self.last_reverse_search_info = {
            "sampled_subtrees": float(len(subtrees)),
            "topk_used": float(len(selected_states)),
            "chosen_delta": float(best_gain if np.isfinite(best_gain) else -1.0),
            "called": float(self.reverse_called_count),
            "passed": float(self.reverse_gate_pass_count),
            "written": float(self.reverse_updates_written_count),
        }
        return best_idx

    def _choose_reverse_search_index(self, state: ExpTree, full_episode: List[str], local_path: List[str], base_reward: float) -> Optional[int]:
        if self.reverse_mode == "prior_anchor":
            return self._apply_prior_anchored_reverse(state, full_episode, local_path, base_reward)
        if not local_path:
            return None
        subtrees = self._enumerate_subtrees(local_path)
        if not subtrees:
            return None

        if self.reverse_search_subtree_samples <= 0:
            sampled = subtrees
        else:
            k = min(self.reverse_search_subtree_samples, len(subtrees))
            sampled = random.sample(subtrees, k)


        const_token = 'R' if 'R' in self.gp_manager.ops else ('x0' if 'x0' in self.gp_manager.ops else local_path[0])
        scored: List[Tuple[float, int, int, List[str], float]] = []
        for idx, size in sampled:
            cf_const = local_path[:idx] + [const_token] + local_path[idx + size:]
            reward_const = self._evaluate_candidate_reward(state, cf_const)
            delta_const = base_reward - reward_const
            if self.reverse_where_metric == "repairability":
                score = max(0.0, reward_const - base_reward)
            else:
                score = delta_const
            scored.append((score, idx, size, cf_const, reward_const))

        scored.sort(key=lambda x: x[0], reverse=True)
        if self.reverse_topk_edges > 0:
            scored = scored[: min(self.reverse_topk_edges, len(scored))]

        best_idx = None
        best_delta = np.inf
        for _where_score, idx, size, cf_const, reward_const in scored:
            delta_const = base_reward - reward_const
            if self.reverse_mode == "where_only":
                self._apply_where_only_feedback(full_episode, local_path, idx, base_reward, reward_const, cf_const)
            else:
                self._apply_contrastive_edge_feedback(full_episode, local_path, idx, const_token, base_reward, reward_const, cf_const)
            if delta_const < best_delta:
                best_delta = delta_const
                best_idx = idx

            if self.reverse_mode in {"ablation", "where_only"}:
                continue


            if self.contrastive_enable_op_replace:
                root_op = local_path[idx]
                arity = self.gp_manager.arity_dict.get(root_op, -1)
                same_arity_ops = [op for op in self.gp_manager.ops if self.gp_manager.arity_dict.get(op, -2) == arity and op != root_op]

                edge_pos = max(0, len(full_episode) - len(local_path)) + idx
                edge_prefix = full_episode[:edge_pos]
                if self.reverse_topm_actions > 0 and same_arity_ops:
                    prior = self.stat_policy.get_action_prior(edge_prefix, same_arity_ops)
                    same_arity_ops = sorted(same_arity_ops, key=lambda a: prior.get(a, 0.0), reverse=True)[: self.reverse_topm_actions]

                for op in same_arity_ops:
                    cf_replace = list(local_path)
                    cf_replace[idx] = op
                    reward_rep = self._evaluate_candidate_reward(state, cf_replace)
                    self._apply_contrastive_edge_feedback(full_episode, local_path, idx, op, base_reward, reward_rep, cf_replace)
                    d = base_reward - reward_rep
                    if d < best_delta:
                        best_delta = d
                        best_idx = idx

        self.last_reverse_search_info = {
            "sampled_subtrees": float(len(sampled)),
            "chosen_delta": float(best_delta if np.isfinite(best_delta) else -1.0),
            "topk_used": float(len(scored)),
            "called": float(self.reverse_called_count),
            "passed": float(self.reverse_gate_pass_count),
            "written": float(self.reverse_updates_written_count),
        }
        return best_idx

    def _perform_mutation(self, node: MCTS_Node, state: ExpTree) -> float:
        try:
            old_path = node.path_queue.random_sample()[0]
            new_path = self.gp_manager.mutate(state, old_path)

            reward, path = self.rollout_once(state, new_path)
            self.count_num += 1
            node.backpropagate(path, reward)
            node.propagate(path, reward)
            self.stat_policy.record_episode(node.get_prefix_path() + list(path), reward)
            return reward
        except Exception:
            return 0.0

    def _perform_crossover(self, node: MCTS_Node, state: ExpTree) -> float:
        path1 = node.path_queue.random_sample()[0]
        path2 = node.path_queue.random_sample()[0]
        new_path1, new_path2 = self.gp_manager.crossover(state, path1, path2)

        rewards: List[float] = []
        for path in [new_path1, new_path2]:
            try:
                reward, final_path = self.rollout_once(state, path)
                self.count_num += 1
                node.backpropagate(final_path, reward)
                node.propagate(final_path, reward)
                self.stat_policy.record_episode(node.get_prefix_path() + list(final_path), reward)
                rewards.append(reward)
            except ValueError:
                rewards.append(0.0)
        return max(rewards) if rewards else 0.0

    def reward(self, state: ExpTree) -> float:
        expression, reward = self.optimizer.optimize_constants(state)
        self.exp_queue.append(expression, reward)
        return reward

    def _sample_guided_rollout_path(self, state: ExpTree) -> List[str]:
        sampled: List[str] = []
        while not state.is_terminal():
            actions = list(state.available_ops)
            if not actions:
                break

            if random.random() < self.rollout_epsilon:
                op = random.choice(actions)
            else:
                prefix = state.op_list
                prior = self.stat_policy.get_action_prior(prefix, actions)
                probs = np.array([prior.get(a, 0.0) for a in actions], dtype=float)
                if probs.sum() <= 1e-12:
                    op = random.choice(actions)
                else:
                    probs = probs ** (1.0 / self.rollout_temperature)
                    probs = probs / probs.sum()
                    op = actions[int(np.random.choice(len(actions), p=probs))]

            state.add_op(op)
            sampled.append(op)

        return sampled

    def rollout_once(self, state: ExpTree, path: Optional[List[str]] = None) -> Tuple[float, List[str]]:
        if path:
            cloned = copy.deepcopy(state)
            for op in path:
                cloned.add_op(op)
            return self.reward(cloned), path

        cloned = copy.deepcopy(state)
        if self.rollout_guided:
            sampled_path = self._sample_guided_rollout_path(cloned)
            return self.reward(cloned), sampled_path

        filled_state, sampled_path = cloned.random_fill()
        return self.reward(filled_state), sampled_path
