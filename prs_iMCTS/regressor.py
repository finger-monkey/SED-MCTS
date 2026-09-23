import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sympy import sympify, expand, expand_log
import time
import random
import copy

from prs_iMCTS.mcts import MCTS
from prs_iMCTS.src import ExpTree, Optimizer, StatisticalPolicyValue, DiversityPool, TwoStageRefiner
from prs_iMCTS.src.utils.gradient_refiner import RefinementEvent
from prs_iMCTS.gp import GPManager


def simplify_expression(exp_str: str, verbose: bool = False) -> str:

    try:
        cleaned = exp_str.replace("[", "").replace("]", "")
        expr = sympify(cleaned)
        expanded = expand(expand_log(expr, force=True))
        return str(expanded)
    except Exception as e:
        if verbose:
            print(f"Simplification failed: {e}")
        return exp_str


class Regressor:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        ops: List[str] = None,
        arity_dict: Dict[str, int] = None,
        context: Dict = None,
        max_depth: int = 6,
        K: int = 500,
        c: float = 4.0,
        gamma: float = 0.5,
        gp_rate: float = 0.2,
        mutation_rate: float = 0.1,
        exploration_rate: float = 0.2,
        max_single_arity_ops: int = 999,
        max_constants: int = 10,
        max_expressions: float = 2e6,
        verbose: bool = False,
        reward_func: Optional[Callable] = None,
        optimization_method: str = 'LD_LBFGS',
        stat_policy_alpha: float = 1.0,
        stat_policy_tau: float = 1.0,
        stat_policy_beta: float = 2.0,
        stat_failure_penalty: float = 2.5,
        stat_failure_reward_threshold: float = 0.05,
        stat_min_samples_for_bias: int = 1,
        stat_risk_lambda: float = 0.5,
        rollout_guided: bool = True,
        rollout_epsilon: float = 0.15,
        rollout_temperature: float = 1.0,
        prior_bootstrap_random_expressions: int = 0,
        prior_bootstrap_topk: int = 0,
        reverse_search_enabled: bool = False,
        reverse_search_subtree_samples: int = 4,
        reverse_search_every: int = 25,
        reverse_mode: str = 'where_only',
        contrastive_enable_op_replace: bool = True,
        contrastive_reward_weight: float = 0.9,
        contrastive_penalty_weight: float = 0.8,
        reverse_topk_edges: int = 0,
        reverse_topm_actions: int = 0,
        reverse_winner_only: bool = True,
        reverse_min_improvement: float = 1e-4,
        reverse_complexity_lambda: float = 0.0,
        reverse_where_metric: str = 'harmfulness',
        reverse_prior_topk_states: int = 3,
        reverse_prior_topm_actions: int = 3,
        reverse_physics_weight: float = 0.7,
        reverse_data_weight: float = 0.3,
        enable_two_stage_refine: bool = False,
        refine_topk_subtrees: int = 3,
        refine_topm_actions: int = 3,
        refine_delta_threshold: float = 1e-4,
        refine_pde_mode: str = 'none',
        refine_advection_beta: float = 1.0,
        refine_physics_lambda: float = 0.0,
        refine_stat_feedback_weight: float = 0.22,
        refine_stat_feedback_max_events: int = 8,
        refine_ablation_feedback_weight: float = 0.15,
        refine_ablation_feedback_max_edges: int = 8,
        refine_every_n: int = 2,
        refine_guided_bootstrap: bool = False,
        refine_guided_bootstrap_warmup_expressions: int = 200,
        refine_guided_bootstrap_affinity: float = 0.55,
        refine_guided_bootstrap_max_buffer: int = 256,
    ):
        self._validate_inputs(x_train, y_train, max_depth)

        self.x_train = x_train
        self.y_train = y_train
        self.verbose = verbose
        self.reward_func = reward_func

        self.ops = self._init_operations(ops, x_train.shape[0])
        self.arity_dict = self._init_arity_dict(arity_dict, x_train.shape[0])
        self.global_context = self._init_context(context)

        self._init_optimization_params(
            max_depth,
            K,
            c,
            gamma,
            gp_rate,
            mutation_rate,
            exploration_rate,
            max_single_arity_ops,
            max_constants,
            max_expressions,
        )

        self.stat_policy = StatisticalPolicyValue(
            alpha=stat_policy_alpha,
            tau=stat_policy_tau,
            beta=stat_policy_beta,
            failure_penalty=stat_failure_penalty,
            failure_reward_threshold=stat_failure_reward_threshold,
            min_samples_for_bias=stat_min_samples_for_bias,
        )
        self.stat_risk_lambda = max(0.0, stat_risk_lambda)
        self.rollout_guided = rollout_guided
        self.rollout_epsilon = rollout_epsilon
        self.rollout_temperature = rollout_temperature

        self.prior_bootstrap_random_expressions = max(0, int(prior_bootstrap_random_expressions))
        self.prior_bootstrap_topk = max(0, int(prior_bootstrap_topk))
        self.reverse_search_enabled = bool(reverse_search_enabled)
        self.reverse_search_subtree_samples = int(reverse_search_subtree_samples)
        self.reverse_search_every = max(1, int(reverse_search_every))
        self.reverse_mode = str(reverse_mode).lower().strip()
        self.contrastive_enable_op_replace = bool(contrastive_enable_op_replace)
        self.contrastive_reward_weight = float(contrastive_reward_weight)
        self.contrastive_penalty_weight = float(contrastive_penalty_weight)
        self.reverse_topk_edges = int(reverse_topk_edges)
        self.reverse_topm_actions = int(reverse_topm_actions)
        self.reverse_winner_only = bool(reverse_winner_only)
        self.reverse_min_improvement = float(reverse_min_improvement)
        self.reverse_complexity_lambda = float(reverse_complexity_lambda)
        self.reverse_where_metric = str(reverse_where_metric).lower().strip()
        self.reverse_prior_topk_states = int(reverse_prior_topk_states)
        self.reverse_prior_topm_actions = int(reverse_prior_topm_actions)
        self.reverse_physics_weight = float(reverse_physics_weight)
        self.reverse_data_weight = float(reverse_data_weight)

        self.enable_two_stage_refine = bool(enable_two_stage_refine)
        self.refine_topk_subtrees = max(1, int(refine_topk_subtrees))
        self.refine_topm_actions = max(1, int(refine_topm_actions))
        self.refine_delta_threshold = float(refine_delta_threshold)
        self.refine_pde_mode = str(refine_pde_mode).lower().strip()
        self.refine_advection_beta = float(refine_advection_beta)
        self.refine_physics_lambda = max(0.0, float(refine_physics_lambda))
        self.refine_stat_feedback_weight = max(0.0, float(refine_stat_feedback_weight))
        self.refine_stat_feedback_max_events = max(0, int(refine_stat_feedback_max_events))
        self.refine_ablation_feedback_weight = max(0.0, float(refine_ablation_feedback_weight))
        self.refine_ablation_feedback_max_edges = max(0, int(refine_ablation_feedback_max_edges))
        self.refine_every_n = max(1, int(refine_every_n))

        self.refine_guided_bootstrap = bool(refine_guided_bootstrap)
        self.refine_guided_bootstrap_warmup_expressions = max(0, int(refine_guided_bootstrap_warmup_expressions))
        self.refine_guided_bootstrap_affinity = float(np.clip(refine_guided_bootstrap_affinity, 0.0, 1.0))
        self.refine_guided_bootstrap_max_buffer = max(8, int(refine_guided_bootstrap_max_buffer))
        self._refine_guided_bootstrap_events: List[RefinementEvent] = []
        self._refine_guided_collecting: bool = False

        self.optimizer = Optimizer(
            x_train,
            y_train,
            self.global_context,
            reward_func,
            optimization_method,
        )

        self.exp_tree = self._create_exp_tree()

        self.diversity_pool = DiversityPool()
        self.refiner = TwoStageRefiner(
            optimizer=self.optimizer,
            stat_policy=self.stat_policy,
            arity_dict=self.arity_dict,
            delta_threshold=self.refine_delta_threshold,
            topk_subtrees=self.refine_topk_subtrees,
            topm_actions=self.refine_topm_actions,
            pde_mode=self.refine_pde_mode,
            advection_beta=self.refine_advection_beta,
            physics_lambda=self.refine_physics_lambda,
            stat_feedback_weight=self.refine_stat_feedback_weight,
            stat_feedback_max_events=self.refine_stat_feedback_max_events,
            ablation_feedback_weight=self.refine_ablation_feedback_weight,
            ablation_feedback_max_edges=self.refine_ablation_feedback_max_edges,
        )

    def _compute_physics_residual(self, expr_str: str) -> float:





        if not hasattr(self, 'refiner') or self.refiner.pde_mode == "none":
            return float('nan')
        try:
            return self.refiner._pde_loss(expr_str)
        except Exception:
            return float('nan')

    def _bootstrap_stat_prior(self) -> None:
        n = self.prior_bootstrap_random_expressions
        if n <= 0:
            return

        candidates: List[Tuple[float, List[str], float]] = []
        if self.verbose:
            print(f"[BOOTSTRAP] Sampling {n} random expressions for prior statistics...")

        for _ in range(n):
            st = copy.deepcopy(self.exp_tree)
            try:
                filled_state, path = st.random_fill()
                expr_opt, reward = self.optimizer.optimize_constants(filled_state)
                data_reward = float(reward) if np.isfinite(reward) else -1.0

                physics_residual = self._compute_physics_residual(expr_opt)
                if np.isfinite(physics_residual):
                    sort_metric = physics_residual
                    use_physics = True
                else:
                    sort_metric = data_reward
                    use_physics = False

                candidates.append((sort_metric, path, use_physics))
            except Exception:
                candidates.append((float('inf'), [], False))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0], reverse=False)
        topk = self.prior_bootstrap_topk if self.prior_bootstrap_topk > 0 else max(1, int(0.2 * len(candidates)))
        topk = min(topk, len(candidates))

        selected = candidates[:topk]
        use_physics_count = sum(1 for _, _, use_phys in selected if use_phys)

        for sort_metric, path, use_physics in selected:
            if path:
                self.stat_policy.record_episode(path, sort_metric)

        if self.verbose:
            best_metric = selected[0][0] if selected else float('nan')
            metric_type = "physics_residual" if selected and selected[0][2] else "data_reward"
            print(f"[BOOTSTRAP] Selected top-{topk}/{len(candidates)} for prior init. "
                  f"best_{metric_type}={best_metric:.6f} ({use_physics_count} physics-based)")

    def _complete_tree_from_prefix_and_op(self, prefix: List[str], next_op: str) -> Optional[ExpTree]:

        st = self._create_exp_tree()
        try:
            for op in prefix:
                st.add_op(op)
            st.add_op(next_op)
            st.random_fill()
            return st
        except Exception:
            return None

    def _bootstrap_stat_prior_refine_guided(self) -> None:





        n = self.prior_bootstrap_random_expressions
        if n <= 0:
            return

        hints = [ev for ev in self._refine_guided_bootstrap_events if ev.improved]
        affinity = self.refine_guided_bootstrap_affinity
        if self.verbose:
            print(
                f"[BOOTSTRAP/refine-guided] Sampling {n} expressions; "
                f"hints={len(hints)}, affinity={affinity:.2f}"
            )

        candidates: List[Tuple[float, List[str], float]] = []
        for _ in range(n):
            st = None
            path: List[str] = []
            if hints and random.random() < affinity:
                ev = random.choice(hints)
                st = self._complete_tree_from_prefix_and_op(list(ev.prefix), ev.action)
            if st is None:
                st = copy.deepcopy(self.exp_tree)
                try:
                    filled_state, path = st.random_fill()
                    st = filled_state
                except Exception:
                    candidates.append((float('inf'), [], False))
                    continue
            try:
                path = list(getattr(st, "op_list", []))
                expr_opt, reward = self.optimizer.optimize_constants(st)
                data_reward = float(reward) if np.isfinite(reward) else -1.0

                physics_residual = self._compute_physics_residual(expr_opt)
                if np.isfinite(physics_residual):
                    sort_metric = physics_residual
                    use_physics = True
                else:
                    sort_metric = data_reward
                    use_physics = False

                candidates.append((sort_metric, path, use_physics))
            except Exception:
                candidates.append((float('inf'), [], False))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0], reverse=False)
        topk = self.prior_bootstrap_topk if self.prior_bootstrap_topk > 0 else max(1, int(0.2 * len(candidates)))
        topk = min(topk, len(candidates))

        selected = candidates[:topk]
        use_physics_count = sum(1 for _, _, use_phys in selected if use_phys)

        for sort_metric, path, use_physics in selected:
            if path:
                self.stat_policy.record_episode(path, sort_metric)

        if self.verbose:
            best_metric = selected[0][0] if selected else float("nan")
            metric_type = "physics_residual" if selected and selected[0][2] else "data_reward"
            print(
                f"[BOOTSTRAP/refine-guided] Selected top-{topk}/{len(candidates)} for prior init. "
                f"best_{metric_type}={best_metric:.6f} ({use_physics_count} physics-based)"
            )

    def fit(self, seed: int = None) -> Tuple[str, float, int, int, List]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        with np.errstate(all='ignore'):
            self._refine_guided_bootstrap_events.clear()
            self._refine_guided_collecting = False

            use_rg = (
                self.refine_guided_bootstrap
                and self.prior_bootstrap_random_expressions > 0
                and self.refine_guided_bootstrap_warmup_expressions > 0
            )
            warmup_limit = min(
                self.refine_guided_bootstrap_warmup_expressions,
                int(self.max_expressions),
            )

            mcts: Optional[MCTS] = None
            saved_enable_refine: Optional[bool] = None
            if use_rg:
                saved_enable_refine = self.enable_two_stage_refine
                if not self.enable_two_stage_refine:
                    self.enable_two_stage_refine = True
                self._refine_guided_collecting = True
                mcts = self._create_mcts()
                self.start_time = time.time()
                self.find_best(mcts, stop_at=warmup_limit)
                self._refine_guided_collecting = False
                self._bootstrap_stat_prior_refine_guided()
                if saved_enable_refine is not None:
                    self.enable_two_stage_refine = saved_enable_refine
            else:
                self._bootstrap_stat_prior()
                mcts = self._create_mcts()
                self.start_time = time.time()

            self.find_best(mcts)
            exp_str, reward = mcts.exp_queue.best()

            best_path = mcts.root.path_queue.best()[0] if mcts.root.path_queue.best()[0] is not None else []
            return (
                simplify_expression(exp_str, self.verbose),
                exp_str,
                mcts.count_num,
                best_path,
            )

    def _build_state_from_path(self, path: List[str]) -> Optional[ExpTree]:
        if not path:
            return None
        st = self._create_exp_tree()
        try:
            for op in path:
                st.add_op(op)
            if not st.is_terminal():
                return None
            return st
        except Exception:
            return None

    def _run_two_stage_refine_if_needed(self, mcts: MCTS, search_reward: float) -> float:
        if not self.enable_two_stage_refine:
            return float(search_reward)



        if self.refine_every_n > 1 and (mcts.count_num % self.refine_every_n) != 0:
            return float(search_reward)

        best_expr, best_reward = mcts.exp_queue.best()
        if best_expr is None or best_reward is None:
            return float(search_reward)

        best_reward = float(best_reward)

        queue_best_before = best_reward
        best_path = mcts.root.path_queue.best()[0] if mcts.root.path_queue.best()[0] is not None else []
        state = self._build_state_from_path(best_path)
        if state is None:
            return float(search_reward)


        stage1_expr, stage1_reward = self.refiner.tune_constants(state)
        self.diversity_pool.upsert(stage1_expr, stage1_reward, best_path, metadata={"stage": "const"})
        promoted = queue_best_before
        if stage1_reward > promoted + 1e-9:
            mcts.exp_queue.append(stage1_expr, stage1_reward)
            promoted = max(promoted, float(stage1_reward))


        stage2_expr, stage2_reward, events, ablation_edges = self.refiner.structural_refine(state, self.ops)
        written = self.refiner.feedback_to_stat_policy(events) + int(ablation_edges)

        if self.refine_guided_bootstrap and self._refine_guided_collecting:
            for ev in events:
                if ev.improved:
                    self._refine_guided_bootstrap_events.append(copy.deepcopy(ev))
                    if len(self._refine_guided_bootstrap_events) > self.refine_guided_bootstrap_max_buffer:
                        self._refine_guided_bootstrap_events.pop(0)

        if stage2_expr is not None and stage2_reward > promoted + 1e-9:
            self.diversity_pool.upsert(stage2_expr, stage2_reward, best_path, metadata={"stage": "struct"})
            mcts.exp_queue.append(stage2_expr, stage2_reward)
            promoted = max(promoted, float(stage2_reward))
            if self.verbose:
                print(
                    f"[REFINE] promoted reward -> {stage2_reward:.6f} "
                    f"(queue was {queue_best_before:.6f}), stat_feedback={written}"
                )
        elif self.verbose and written > 0:
            print(f"[REFINE] queue/promoted={promoted:.6f} (was {queue_best_before:.6f}), stat_feedback={written}")

        _, q_best = mcts.exp_queue.best()
        if q_best is not None and np.isfinite(float(q_best)):
            return max(float(search_reward), float(q_best))
        return max(float(search_reward), float(promoted))

    def find_best(self, mcts: MCTS, stop_at: Optional[int] = None):
        last_report_time = getattr(self, '_last_report_time', self.start_time)
        REPORT_INTERVAL = 10.0
        last_seen_best = -np.inf
        limit = int(self.max_expressions) if stop_at is None else min(int(stop_at), int(self.max_expressions))

        while mcts.count_num < limit:
            search_r = mcts.search(self.exp_tree)
            best_reward = self._run_two_stage_refine_if_needed(mcts, search_r)
            last_seen_best = max(last_seen_best, float(best_reward))

            now = time.time()
            if self.verbose and (now - last_report_time >= REPORT_INTERVAL):
                self.print_status(mcts)
                last_report_time = now
                self._last_report_time = last_report_time

            if now - self.start_time > 172800:
                if self.verbose:
                    self.print_status(mcts)
                break

            if 1 - best_reward < mcts.succ_error_tol:
                if self.verbose:
                    self.print_status(mcts)
                break

    def predict(self, x: np.ndarray, vec_exp_str) -> np.ndarray:
        try:
            f_pred = eval(f'lambda x: {vec_exp_str}', self.optimizer.context)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                y = f_pred(x)
            return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def print_status(self, mcts) -> None:
        try:
            best_expr, best_reward = mcts.exp_queue.best()
            f_pred = eval(f'lambda x: {best_expr}', self.optimizer.context)
            y_pred = f_pred(self.x_train)
            y_true = self.y_train
            elapsed_time = time.time() - self.start_time

            with np.errstate(divide='ignore', invalid='ignore'):
                ss_res = np.sum((y_pred - y_true) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
                nmse = np.mean((y_pred - y_true) ** 2) / np.var(y_true) if np.var(y_true) != 0 else float('nan')

            report = [
                "\n\033[1;36m=== Symbolic Regression Progress Report ===\033[0m",
                f"\033[1mEvaluated Expressions:\033[0m {mcts.count_num}",
                f"\033[1mElapsed Time:\033[0m {elapsed_time:.1f}s",
                "\n\033[1mTop Performance Metrics:\033[0m",
                f"  \033[32mBest Expression:\033[0m {simplify_expression(best_expr, self.verbose)}",
                f"  \033[33mReward (↑):\033[0m {best_reward:.3e}",
                f" R² (↑): {r_squared:.4f} | NMSE (↓): {nmse:.3e}",
                "\n\033[1mExploration Profile:\033[0m",
                f"  Total Nodes: {mcts.total_nodes} | Active Branches: {len(mcts.root.children)}",
                f"  Exploration Rate: {self.exploration_rate:.2f} | Mutation Rate: {self.mutation_rate:.2f}",
            ]

            if mcts.root.children:
                report.append("\n\033[1mBranch Statistics:\033[0m")
                header = f"{'Operator':<10} {'Visits':<8} {'Best Reward':<12} {'Dominance':<10}"
                report.append("\033[34m" + header + "\033[0m")

                for child in sorted(mcts.root.children, key=lambda x: x.visits, reverse=True):
                    reward = child.path_queue.best()[1]
                    dominance = child.visits / mcts.root.visits
                    report.append(
                        f"{child.move:<10} {child.visits:<8} {reward:.3e} {dominance:>7.1%}"
                    )

            print("\n".join(report))
            print("\033[1;35m" + "═" * 60 + "\033[0m")

        except Exception as e:
            print(f"\033[31mStatus Update Error: {str(e)}\033[0m")

    def _validate_inputs(self, x_train, y_train, max_depth):
        if x_train.shape[1] != y_train.shape[0]:
            raise ValueError("Mismatched dimensions between x_train and y_train")

        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

    def _init_operations(self, ops, var_count) -> List[str]:
        default_ops = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
        ops = ops or default_ops
        return ops + [f'x{i}' for i in range(var_count)]

    def _init_arity_dict(self, arity_dict, var_count) -> Dict[str, int]:
        default_arity = {
            '+': 2, '-': 2, '*': 2, '/': 2,
            'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'tanh': 1,
            'C': 0, 'R': 0,
        }
        arity_dict = arity_dict or default_arity.copy()
        arity_dict.update({f'x{i}': 0 for i in range(var_count)})
        return arity_dict

    def _init_context(self, context) -> Dict:
        default_context = {
            'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'log': np.log, 'tanh': np.tanh,
        }
        return {**default_context, **(context or {})}

    def _init_optimization_params(self, max_depth, K, c, gamma, gp_rate,
                                 mutation_rate, exploration_rate,
                                 max_single_arity_ops, max_constants, max_expressions):
        self.max_depth = max_depth
        self.K = K
        self.c = c
        self.gamma = gamma
        self.gp_rate = np.clip(gp_rate, 0.0, 1.0)
        self.mutation_rate = np.clip(mutation_rate, 0.0, 1.0)
        self.exploration_rate = np.clip(exploration_rate, 0.0, 1.0)
        self.max_single_arity_ops = max_single_arity_ops
        self.max_constants = max_constants
        self.max_expressions = max_expressions

    def _create_exp_tree(self):
        return ExpTree(
            max_depth=self.max_depth,
            max_single_arity_ops=self.max_single_arity_ops,
            max_constants=self.max_constants,
            arity_dict=self.arity_dict,
            ops=self.ops,
        )

    def _create_mcts(self):
        return MCTS(
            optimizer=self.optimizer,
            gp_manager=GPManager(self.ops, self.arity_dict),
            gp_rate=self.gp_rate,
            mutation_rate=self.mutation_rate,
            exploration_rate=self.exploration_rate,
            K=self.K,
            c=self.c,
            gamma=self.gamma,
            verbose=self.verbose,
            stat_policy=self.stat_policy,
            risk_lambda=self.stat_risk_lambda,
            rollout_guided=self.rollout_guided,
            rollout_epsilon=self.rollout_epsilon,
            rollout_temperature=self.rollout_temperature,
            reverse_search_enabled=self.reverse_search_enabled,
            reverse_search_subtree_samples=self.reverse_search_subtree_samples,
            reverse_search_every=self.reverse_search_every,
            reverse_mode=self.reverse_mode,
            contrastive_enable_op_replace=self.contrastive_enable_op_replace,
            contrastive_reward_weight=self.contrastive_reward_weight,
            contrastive_penalty_weight=self.contrastive_penalty_weight,
            reverse_topk_edges=self.reverse_topk_edges,
            reverse_topm_actions=self.reverse_topm_actions,
            reverse_winner_only=self.reverse_winner_only,
            reverse_min_improvement=self.reverse_min_improvement,
            reverse_complexity_lambda=self.reverse_complexity_lambda,
            reverse_where_metric=self.reverse_where_metric,
            reverse_prior_topk_states=self.reverse_prior_topk_states,
            reverse_prior_topm_actions=self.reverse_prior_topm_actions,
            reverse_physics_weight=self.reverse_physics_weight,
            reverse_data_weight=self.reverse_data_weight,
        )
