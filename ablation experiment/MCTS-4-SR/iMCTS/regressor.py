import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sympy import sympify, expand, expand_log
import time
import random
import copy

from iMCTS.mcts import MCTS
from iMCTS.src import ExpTree, Optimizer, StatisticalPolicyValue
from iMCTS.gp import GPManager


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
        optimization_method: str = 'LN_NELDERMEAD',
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

        self.optimizer = Optimizer(
            x_train,
            y_train,
            self.global_context,
            reward_func,
            optimization_method,
        )

        self.exp_tree = self._create_exp_tree()

    def _bootstrap_stat_prior(self) -> None:
        n = self.prior_bootstrap_random_expressions
        if n <= 0:
            return

        candidates: List[Tuple[float, List[str]]] = []
        if self.verbose:
            print(f"[BOOTSTRAP] Sampling {n} random expressions for prior statistics...")

        for _ in range(n):
            st = copy.deepcopy(self.exp_tree)
            try:
                filled_state, path = st.random_fill()
                _, reward = self.optimizer.optimize_constants(filled_state)
                r = float(reward) if np.isfinite(reward) else -1.0
                candidates.append((r, path))
            except Exception:
                candidates.append((-1.0, []))

        candidates.sort(key=lambda x: x[0], reverse=True)
        topk = self.prior_bootstrap_topk if self.prior_bootstrap_topk > 0 else max(1, int(0.2 * len(candidates)))
        topk = min(topk, len(candidates))

        selected = candidates[:topk]
        for reward, path in selected:
            if path:
                self.stat_policy.record_episode(path, reward)

        if self.verbose:
            best_r = selected[0][0] if selected else float('nan')
            print(f"[BOOTSTRAP] Selected top-{topk}/{len(candidates)} for prior init. best_reward={best_r:.6f}")

    def fit(self, seed: int = None) -> Tuple[str, float, int, int, List]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        with np.errstate(all='ignore'):
            self._bootstrap_stat_prior()
            mcts = self._create_mcts()
            self.start_time = time.time()
            self.find_best(mcts)
            exp_str, reward = mcts.exp_queue.best()

            return (
                simplify_expression(exp_str, self.verbose),
                exp_str,
                mcts.count_num,
                mcts.root.path_queue.best()[0],
            )

    def find_best(self, mcts: MCTS):
        last_report_time = getattr(self, '_last_report_time', self.start_time)
        REPORT_INTERVAL = 10.0
        while mcts.count_num < self.max_expressions:
            best_reward = mcts.search(self.exp_tree)

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
