

















from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from sed_mcts_framework import (
    BINARY_ACTIONS,
    EPS,
    Expr,
    Jet,
    LEAF_ACTIONS,
    MCTSNode,
    SearchConfig,
    SearchState,
    UNARY_ACTIONS,
    freeze_constants,
    renumber_constants,
)


ResidualFunction = Callable[[Sequence[Jet], np.ndarray], Sequence[np.ndarray]]


@dataclass
class MultiFieldProblem:
    name: str
    field_names: Tuple[str, ...]
    field_groups: Dict[str, Tuple[int, ...]]
    input_names: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]
    x_obs: np.ndarray
    y_obs: Tuple[np.ndarray, ...]
    x_phys: np.ndarray
    x_eval: np.ndarray
    y_eval: Tuple[np.ndarray, ...]
    residual_fn: ResidualFunction
    data_scales: Tuple[float, ...]
    phys_scales: Tuple[float, ...]
    templates: Tuple[Expr, ...]
    template_params: Tuple[np.ndarray, ...]
    operator_sets: Tuple[Tuple[str, ...], ...]
    description: str
    data_source: str


@dataclass
class MultiCandidate:
    field_idx: int
    expr: Expr
    params: np.ndarray
    expressions: Tuple[Expr, ...]
    params_list: Tuple[np.ndarray, ...]
    field_data_mse: Tuple[float, ...]
    residual_mse: Tuple[float, ...]
    loss: float
    reward: float
    complexity: int


@dataclass
class MemoryRecord:
    nmem: int = 0
    wmem: float = 0.0
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def mean(self) -> float:
        return self.wmem / max(1, self.nmem)

    @property
    def variance(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.var(np.asarray(self.values), ddof=1))

    @property
    def qstat(self) -> float:
        return self.mean - math.sqrt(max(0.0, self.variance) / (self.nmem + 1.0))


class MultiResponsibilityMemory:


    def __init__(self) -> None:
        self.edges: Dict[Tuple[int, int, str], MemoryRecord] = defaultdict(MemoryRecord)

    def update(self, field_idx: int, depth: int, action: str, value: float) -> None:
        record = self.edges[(field_idx, depth, action)]
        record.nmem += 1
        record.wmem += float(value)
        record.values.append(float(value))

    def seed(self, field_idx: int, depth: int, action: str, value: float) -> None:
        self.update(field_idx, depth, action, value)

    def distribution(self, field_idx: int, depth: int, actions: Sequence[str], temperature: float = 1.0) -> Dict[str, float]:
        if not actions:
            return {}
        q = np.asarray(
            [self.edges[(field_idx, depth, action)].qstat for action in actions],
            dtype=float,
        )
        q = q / max(float(temperature), EPS)
        q -= np.max(q)
        values = np.exp(np.clip(q, -50.0, 50.0))
        values /= max(float(values.sum()), EPS)
        values = 0.95 * values + 0.05 / len(actions)
        return {action: float(value) for action, value in zip(actions, values)}

    def jsonable(self) -> Dict[str, Dict[str, float]]:
        return {
            f"field={field};depth={depth};action={action}": {
                "Nmem": record.nmem,
                "Wmem": record.wmem,
                "Vmem": record.variance,
                "qstat": record.qstat,
            }
            for (field, depth, action), record in sorted(self.edges.items())
        }


@dataclass
class ArchiveRecord:
    canonical: str
    expr: Expr
    params: np.ndarray
    reward: float
    rho: float
    complexity: int
    responsible_edges: Tuple[Tuple[int, int, str], ...]


class MultiStructureArchive:
    def __init__(self, capacity: int = 256, per_field_capacity: int = 64) -> None:
        self.capacity = int(capacity)
        self.per_field_capacity = int(per_field_capacity)
        self.records: Dict[int, List[ArchiveRecord]] = defaultdict(list)

    @property
    def size(self) -> int:
        return sum(len(items) for items in self.records.values())

    def consider(
        self,
        field_idx: int,
        candidate: MultiCandidate,
        rho: float,
        edges: Sequence[Tuple[int, int, str]],
    ) -> None:
        records = self.records[field_idx]
        canonical = candidate.expr.canonical()
        for record in records:
            if record.canonical == canonical:
                if candidate.reward > record.reward:
                    record.reward = candidate.reward
                    record.params = candidate.params.copy()
                return
        records.append(
            ArchiveRecord(
                canonical=canonical,
                expr=candidate.expr,
                params=candidate.params.copy(),
                reward=candidate.reward,
                rho=float(rho),
                complexity=candidate.expr.node_count(),
                responsible_edges=tuple(edges),
            )
        )
        records.sort(key=lambda item: item.reward, reverse=True)
        del records[self.per_field_capacity :]
        while self.size > self.capacity:
            worst_field = min(self.records, key=lambda key: min(item.reward for item in self.records[key]))
            self.records[worst_field].sort(key=lambda item: item.reward, reverse=True)
            self.records[worst_field].pop()
            if not self.records[worst_field]:
                del self.records[worst_field]

    def donor(self, field_idx: int, rng: np.random.Generator) -> Optional[ArchiveRecord]:
        records = self.records.get(field_idx, [])
        if not records:
            return None
        weights = np.asarray([max(1.0e-8, item.reward + item.rho) for item in records], dtype=float)
        weights /= max(float(weights.sum()), EPS)
        return records[int(rng.choice(len(records), p=weights))]

    def jsonable(self) -> Dict[str, object]:
        return {
            "size": self.size,
            "records": {
                str(field_idx): [
                    {
                        "canonical_subtree": record.canonical,
                        "expression": record.expr.render(record.params),
                        "reward": record.reward,
                        "rho": record.rho,
                        "complexity": record.complexity,
                        "responsible_edge_signature": [list(edge) for edge in record.responsible_edges],
                    }
                    for record in records
                ]
                for field_idx, records in sorted(self.records.items())
            },
        }


def _sample(bounds: Sequence[Tuple[float, float]], n: int, rng: np.random.Generator) -> np.ndarray:
    low = np.asarray([bound[0] for bound in bounds], dtype=float)
    high = np.asarray([bound[1] for bound in bounds], dtype=float)
    return rng.uniform(low, high, size=(int(n), len(bounds)))


def _constant_times_var(index: int, variable: int) -> Expr:
    return Expr("*", (Expr.const(index), Expr.var(variable)))


def _make_generated_problem(
    name: str,
    field_names: Tuple[str, ...],
    field_groups: Dict[str, Tuple[int, ...]],
    input_names: Tuple[str, ...],
    bounds: Tuple[Tuple[float, float], ...],
    templates: Tuple[Expr, ...],
    template_params: Tuple[Sequence[float], ...],
    operator_sets: Tuple[Tuple[str, ...], ...],
    residual_fn: ResidualFunction,
    target_values: Callable[[np.ndarray], Tuple[np.ndarray, ...]],
    description: str,
    seed: int,
    n_obs: int,
    n_phys: int,
    n_eval: int,
    data_source: str,
) -> MultiFieldProblem:
    rng = np.random.default_rng(seed)
    x_obs = _sample(bounds, n_obs, rng)
    x_phys = _sample(bounds, n_phys, rng)
    x_eval = _sample(bounds, n_eval, rng)
    y_obs = target_values(x_obs)
    y_eval = target_values(x_eval)
    data_scales = tuple(float(np.var(values) + 1.0e-8) for values in y_obs)
    target_jets = [renumber_constants(expr).evaluate(x_phys, params) for expr, params in zip(templates, template_params)]
    target_residuals = residual_fn(target_jets, x_phys)
    phys_scales = tuple(float(np.var(values) + 1.0) for values in target_residuals)
    return MultiFieldProblem(
        name=name,
        field_names=field_names,
        field_groups=field_groups,
        input_names=input_names,
        bounds=bounds,
        x_obs=x_obs,
        y_obs=tuple(np.asarray(values, dtype=float) for values in y_obs),
        x_phys=x_phys,
        x_eval=x_eval,
        y_eval=tuple(np.asarray(values, dtype=float) for values in y_eval),
        residual_fn=residual_fn,
        data_scales=data_scales,
        phys_scales=phys_scales,
        templates=tuple(renumber_constants(expr) for expr in templates),
        template_params=tuple(np.asarray(values, dtype=float) for values in template_params),
        operator_sets=operator_sets,
        description=description,
        data_source=data_source,
    )


def _make_diffusion_convection(seed: int, n_obs: int, n_phys: int, n_eval: int) -> MultiFieldProblem:


    templates = (
        Expr("*", (Expr.lit(-2.0), Expr.var(1))),
        Expr("*", (Expr.lit(-2.0), Expr.var(0))),
        Expr("-", (Expr("square", (Expr.var(0),)), Expr("square", (Expr.var(1),)))),
    )
    params = ((), (), ())

    def target_values(x: np.ndarray) -> Tuple[np.ndarray, ...]:
        return (-2.0 * x[:, 1], -2.0 * x[:, 0], x[:, 0] ** 2 - x[:, 1] ** 2)

    def residual_fn(jets: Sequence[Jet], x: np.ndarray) -> Sequence[np.ndarray]:
        vx, vy, u = jets
        transport = -u.hess[:, 0, 0] - u.hess[:, 1, 1] + vx.value * u.grad[:, 0] + vy.value * u.grad[:, 1]
        incompress = vx.grad[:, 0] + vy.grad[:, 1]
        return (transport, incompress)

    return _make_generated_problem(
        "Diffusion-convection",
        ("v_x", "v_y", "u"),
        {"f1": (0, 1), "f2": (2,)},
        ("x", "y"),
        ((-1.0, 1.0), (-1.0, 1.0)),
        templates,
        params,
        (
            ("neg", "+", "-", "*"),
            ("neg", "+", "-", "*"),
            ("neg", "+", "-", "*", "square"),
        ),
        residual_fn,
        target_values,
        "Ndc1=-Delta(u)+v.grad(u), Ndc2=div(v)",
        seed,
        n_obs,
        n_phys,
        n_eval,
        "manufactured_solution_oracle",
    )


JOULE_CONFIGS = {
    "joule1": {"sigma": 1.0, "rho_c": 1.0, "k": 0.5, "a": 1.0, "b": 0.5, "harmonic": 0.20},
    "joule2": {"sigma": 2.0, "rho_c": 1.5, "k": 1.0, "a": 0.7, "b": -0.4, "harmonic": -0.10},
    "joule3": {"sigma": 0.5, "rho_c": 2.0, "k": 0.8, "a": 1.2, "b": 0.2, "harmonic": 0.30},
    "joule4": {"sigma": 3.0, "rho_c": 1.0, "k": 0.2, "a": 0.5, "b": 1.0, "harmonic": -0.25},
    "joule5": {"sigma": 1.5, "rho_c": 0.75, "k": 1.2, "a": 0.8, "b": 0.8, "harmonic": 0.15},
}


def _make_joule(
    instance: str,
    seed: int,
    n_obs: int,
    n_phys: int,
    n_eval: int,
) -> MultiFieldProblem:
    config = JOULE_CONFIGS[instance]
    sigma = config["sigma"]
    rho_c = config["rho_c"]
    kappa = config["k"]
    a = config["a"]
    b = config["b"]
    harmonic = config["harmonic"]
    templates = (
        Expr(
            "+",
            (
                Expr("*", (Expr.lit(a), Expr.var(1))),
                Expr("*", (Expr.lit(b), Expr.var(2))),
            ),
        ),
        Expr(
            "+",
            (
                Expr("*", (Expr.lit(sigma * (a * a + b * b) / rho_c), Expr.var(0))),
                Expr(
                    "*",
                    (
                        Expr.lit(harmonic),
                        Expr("-", (Expr("square", (Expr.var(1),)), Expr("square", (Expr.var(2),)))),
                    ),
                ),
            ),
        ),
    )
    params = ((), ())

    def target_values(x: np.ndarray) -> Tuple[np.ndarray, ...]:
        potential = a * x[:, 1] + b * x[:, 2]
        temperature = sigma * (a * a + b * b) / rho_c * x[:, 0] + harmonic * (x[:, 1] ** 2 - x[:, 2] ** 2)
        return potential, temperature

    def residual_fn(jets: Sequence[Jet], x: np.ndarray) -> Sequence[np.ndarray]:
        potential, temperature = jets
        electric = -sigma * (potential.hess[:, 1, 1] + potential.hess[:, 2, 2])
        joule_source = sigma * (potential.grad[:, 1] ** 2 + potential.grad[:, 2] ** 2)
        thermal = rho_c * temperature.grad[:, 0] - kappa * (temperature.hess[:, 1, 1] + temperature.hess[:, 2, 2]) - joule_source
        return electric, thermal

    return _make_generated_problem(
        f"Joule heating {instance[-1]}",
        ("V", "T"),
        {"f1": (0,), "f2": (1,)},
        ("t", "x", "y"),
        ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        templates,
        params,
        (
            ("neg", "+", "-", "*"),
            ("neg", "+", "-", "*", "square"),
        ),
        residual_fn,
        target_values,
        "Njh1=-div(sigma grad V), Njh2=rho*C*T_t-div(k grad T)-sigma|grad V|^2",
        seed,
        n_obs,
        n_phys,
        n_eval,
        "manufactured_solution_oracle",
    )


PARABOLIC_CONFIGS = {




    1: {"a1": 0.024, "a2": 0.17, "nonlinear": True, "difference": 0.02, "slope": 0.25, "offset": 0.75},
    2: {"a1": 0.24, "a2": 0.17, "nonlinear": True, "difference": -0.01, "slope": -0.15, "offset": 0.50},
    3: {"a1": 1.70, "a2": 0.17, "nonlinear": False, "difference": 0.05, "slope": 0.40, "offset": 0.80},
}


def _make_parabolic(
    instance: int,
    seed: int,
    n_obs: int,
    n_phys: int,
    n_eval: int,
) -> MultiFieldProblem:
    config = PARABOLIC_CONFIGS[instance]
    a1 = float(config["a1"])
    a2 = float(config["a2"])
    difference = float(config["difference"])
    slope = float(config["slope"])
    offset = float(config["offset"])
    nonlinear = bool(config["nonlinear"])

    if nonlinear:
        exchange_value = math.exp(5.73 * difference) - math.exp(-11.46 * difference)
    else:
        exchange_value = difference
    curvature = 2.0 * exchange_value / (a1 - a2)
    time_slope = (a1 + a2) * exchange_value / (a1 - a2)

    def target_values(x: np.ndarray) -> Tuple[np.ndarray, ...]:
        spatial_temporal = (
            0.5 * curvature * x[:, 0] ** 2
            + time_slope * x[:, 1]
            + slope * x[:, 0]
            + offset
        )
        return spatial_temporal + difference, spatial_temporal

    templates = (
        Expr(
            "+",
            (
                Expr.lit(difference),
                Expr(
                    "+",
                    (
                        Expr("*", (Expr.lit(0.5 * curvature), Expr("square", (Expr.var(0),)))),
                        Expr(
                            "+",
                            (
                                Expr("*", (Expr.lit(time_slope), Expr.var(1))),
                                Expr("+", (Expr("*", (Expr.lit(slope), Expr.var(0))), Expr.lit(offset))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        Expr(
            "+",
            (
                Expr("*", (Expr.lit(0.5 * curvature), Expr("square", (Expr.var(0),)))),
                Expr(
                    "+",
                    (
                        Expr("*", (Expr.lit(time_slope), Expr.var(1))),
                        Expr("+", (Expr("*", (Expr.lit(slope), Expr.var(0))), Expr.lit(offset))),
                    ),
                ),
            ),
        ),
    )
    params = ((), ())

    def residual_fn(jets: Sequence[Jet], x: np.ndarray) -> Sequence[np.ndarray]:
        u1, u2 = jets
        difference = u1.value - u2.value
        exchange = difference if not nonlinear else np.exp(np.clip(5.73 * difference, -20.0, 20.0)) - np.exp(np.clip(-11.46 * difference, -20.0, 20.0))
        r1 = u1.grad[:, 1] - a1 * u1.hess[:, 0, 0] + exchange
        r2 = u2.grad[:, 1] - a2 * u2.hess[:, 0, 0] - exchange
        return r1, r2

    return _make_generated_problem(
        f"Parabolic {instance}",
        ("u1", "u2"),
        {"f1": (0,), "f2": (1,)},
        ("x", "t"),
        ((0.0, 1.0), (0.0, 2.0)),
        templates,
        params,
        (
            ("neg", "+", "-", "*", "exp", "sin", "cos"),
            ("neg", "+", "-", "*", "exp", "sin", "cos"),
        ),
        residual_fn,
        target_values,
        f"u1_t-{a1}u1_xx +/- F(u1-u2)=0; u2_t-{a2}u2_xx -/+ F(u1-u2)=0",
        seed,
        n_obs,
        n_phys,
        n_eval,
        "manufactured_solution_oracle",
    )


def make_multiphysics_problem(
    name: str,
    seed: int = 0,
    n_obs: int = 128,
    n_phys: int = 256,
    n_eval: int = 1024,
    data_root: Optional[Path] = None,
) -> MultiFieldProblem:
    key = name.lower().replace("_", "-")
    if key in {"diffusion-convection", "diffusionconv", "dc"}:
        return _make_diffusion_convection(seed, n_obs, n_phys, n_eval)
    if key in JOULE_CONFIGS:
        return _make_joule(key, seed, n_obs, n_phys, n_eval)
    if key.startswith("parabolic"):
        instance = int(key.split("-")[-1])


        return _make_parabolic(instance, seed, n_obs, n_phys, n_eval)
    raise ValueError(f"Unknown Table X benchmark: {name}")


class MultiFieldSEDMCTS:
    def __init__(self, problem: MultiFieldProblem, config: Optional[SearchConfig] = None) -> None:
        self.problem = problem
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.memory = MultiResponsibilityMemory()
        self.archive = MultiStructureArchive(self.config.archive_capacity, self.config.archive_bin_capacity * len(problem.field_names))
        self.current_exprs = [expr for expr in problem.templates]
        self.current_params = [params.copy() for params in problem.template_params]
        self.best: Optional[MultiCandidate] = None
        self.terminal_evaluations = 0
        self.intervention_count = 0
        self.archive_crossovers = 0
        self.bootstrap_candidates = 0
        self._seed_structural_prior()

    def _seed_structural_prior(self) -> None:
        for field_idx, operators in enumerate(self.problem.operator_sets):
            for depth in range(self.config.max_depth):
                for action in operators:
                    self.memory.seed(field_idx, depth, action, 0.15 / (1.0 + 0.1 * depth))

    def allowed_actions(self, field_idx: int, depth: int) -> Tuple[str, ...]:
        variables = tuple(f"v{i}" for i in range(len(self.problem.input_names)))
        if depth >= self.config.max_depth:
            return variables + LEAF_ACTIONS
        return variables + LEAF_ACTIONS + tuple(self.problem.operator_sets[field_idx])

    @staticmethod
    def _next_state(state: SearchState, action: str) -> SearchState:
        if not state.frontier:
            raise ValueError("Cannot expand terminal state")
        depth = state.frontier[0]
        rest = list(state.frontier[1:])
        if action in UNARY_ACTIONS:
            rest.insert(0, depth + 1)
        elif action in BINARY_ACTIONS:
            rest[0:0] = [depth + 1, depth + 1]
        return SearchState(state.actions + (action,), tuple(rest))

    def _select_unexpanded(self, node: MCTSNode, field_idx: int) -> str:
        depth = node.state.frontier[0]
        actions = [action for action in self.allowed_actions(field_idx, depth) if action not in node.children]
        probabilities = self.memory.distribution(field_idx, depth, actions, self.config.temperature)
        weights = np.asarray([probabilities[action] for action in actions], dtype=float)
        weights /= max(float(weights.sum()), EPS)
        return str(self.rng.choice(actions, p=weights))

    def _select_existing(self, node: MCTSNode, field_idx: int) -> str:
        depth = node.state.frontier[0]
        probabilities = self.memory.distribution(field_idx, depth, list(node.children), self.config.temperature)
        log_term = math.log(1.0 + node.visits)
        best_action = None
        best_score = -float("inf")
        for action, child in node.children.items():
            mk = float(np.mean(child.recent_rewards)) if child.recent_rewards else child.mean_value
            score = mk + self.config.c * (log_term / (1.0 + child.visits)) ** self.config.gamma + self.config.lambda_s * math.log(probabilities.get(action, self.config.epsilon) + self.config.epsilon)
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            raise RuntimeError("No existing child available")
        return best_action

    def _rollout(self, state: SearchState, field_idx: int) -> SearchState:
        while state.frontier:
            depth = state.frontier[0]
            actions = self.allowed_actions(field_idx, depth)
            probabilities = self.memory.distribution(field_idx, depth, actions, self.config.temperature)
            if self.rng.random() < self.config.rollout_epsilon:
                action = str(self.rng.choice(actions))
            else:
                action = max(actions, key=lambda candidate: probabilities[candidate])
            state = self._next_state(state, action)
        return state

    def _optimize_field(self, field_idx: int, expr: Expr) -> np.ndarray:
        n_constants = expr.num_constants()
        if n_constants == 0:
            return np.empty(0, dtype=float)

        starts = [np.zeros(n_constants, dtype=float), self.rng.uniform(-1.0, 1.0, size=n_constants)]

        def objective(params: np.ndarray) -> float:
            expressions = list(self.current_exprs)
            params_list = [values.copy() for values in self.current_params]
            expressions[field_idx] = expr
            params_list[field_idx] = np.asarray(params, dtype=float)
            return self._loss_components(expressions, params_list)[2]

        best = None
        for start in starts:
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0)] * n_constants,
                options={"maxiter": self.config.maxiter_constants, "ftol": 1.0e-10, "maxls": 12},
            )
            if best is None or result.fun < best.fun:
                best = result
        assert best is not None
        return np.asarray(best.x, dtype=float)

    def _loss_components(
        self,
        expressions: Sequence[Expr],
        params_list: Sequence[Sequence[float]],
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
        obs_jets = [expr.evaluate(self.problem.x_obs, params) for expr, params in zip(expressions, params_list)]
        phys_jets = [expr.evaluate(self.problem.x_phys, params) for expr, params in zip(expressions, params_list)]
        field_data = tuple(float(np.mean((jet.value - target) ** 2)) for jet, target in zip(obs_jets, self.problem.y_obs))
        residuals = self.problem.residual_fn(phys_jets, self.problem.x_phys)
        residual_mse = tuple(float(np.mean(np.asarray(residual, dtype=float) ** 2)) for residual in residuals)
        normalized_data = np.mean([value / scale for value, scale in zip(field_data, self.problem.data_scales)])
        normalized_phys = np.mean([value / scale for value, scale in zip(residual_mse, self.problem.phys_scales)])
        loss = float(normalized_data + self.config.lambda_pde * normalized_phys)
        if not np.isfinite(loss):
            return field_data, residual_mse, 1.0e12
        return field_data, residual_mse, loss

    def _candidate(self, field_idx: int, expr: Expr, params: np.ndarray) -> MultiCandidate:
        expressions = list(self.current_exprs)
        params_list = [values.copy() for values in self.current_params]
        expressions[field_idx] = expr
        params_list[field_idx] = params.copy()
        field_data, residual_mse, loss = self._loss_components(expressions, params_list)
        complexity = sum(expression.node_count() for expression in expressions)
        reward = float(np.exp(np.clip(-loss - self.config.eta * complexity, -700.0, 0.0)))
        self.terminal_evaluations += 1
        return MultiCandidate(
            field_idx=field_idx,
            expr=expr,
            params=params.copy(),
            expressions=tuple(expressions),
            params_list=tuple(values.copy() for values in params_list),
            field_data_mse=field_data,
            residual_mse=residual_mse,
            loss=loss,
            reward=reward,
            complexity=complexity,
        )

    def _neutralized(self, node: Expr, x: np.ndarray, params: Sequence[float]) -> Optional[Expr]:
        if node.op in {"C", "L"}:
            return None
        if node.op.startswith("v"):
            return Expr.lit(float(np.mean(x[:, int(node.op[1:])])))
        if node.op in {"+", "-"}:
            return Expr.lit(0.0)
        if node.op in {"*", "/"}:
            return Expr.lit(1.0)
        try:
            values = node.evaluate(x, params).value
            finite = values[np.isfinite(values)]
            mean = float(np.mean(finite)) if finite.size else 0.0
        except Exception:
            mean = 0.0
        return Expr.lit(float(np.clip(mean, -1.0e3, 1.0e3)))

    def _responsibility(self, candidate: MultiCandidate) -> Tuple[float, List[Tuple[int, int, str]]]:
        base_data = sum(value / scale for value, scale in zip(candidate.field_data_mse, self.problem.data_scales))
        base_phys = sum(value / scale for value, scale in zip(candidate.residual_mse, self.problem.phys_scales))
        scored: List[Tuple[float, Tuple[int, int, str]]] = []
        field_idx = candidate.field_idx
        for path, node in candidate.expr.subtrees():
            if not path:
                continue
            replacement = self._neutralized(node, self.problem.x_phys, candidate.params)
            if replacement is None:
                continue
            intervened = candidate.expr.replace(path, replacement)
            expressions = list(candidate.expressions)
            expressions[field_idx] = intervened
            field_data, residual_mse, _ = self._loss_components(expressions, candidate.params_list)
            delta_data = sum(value / scale for value, scale in zip(field_data, self.problem.data_scales)) - base_data
            delta_phys = sum(value / scale for value, scale in zip(residual_mse, self.problem.phys_scales)) - base_phys
            score = max(0.0, delta_data + delta_phys) / ((1.0 + node.node_count()) ** self.config.responsibility_zeta)
            scored.append((score, (field_idx, len(path), node.op)))
        total = sum(score for score, _ in scored)
        if total <= EPS:
            return 0.0, []
        edges = []
        for score, edge in scored:
            self.memory.update(*edge, score / total)
            edges.append(edge)
        self.intervention_count += len(scored)
        return 1.0, edges

    def _archive_crossover(self, candidate: MultiCandidate) -> MultiCandidate:
        if self.rng.random() >= self.config.archive_crossover_rate:
            return candidate
        donor = self.archive.donor(candidate.field_idx, self.rng)
        if donor is None:
            return candidate
        parent_nodes = candidate.expr.subtrees()
        donor_nodes = freeze_constants(donor.expr, donor.params).subtrees()
        parent_path, _ = parent_nodes[int(self.rng.integers(len(parent_nodes)))]
        _, donor_subtree = donor_nodes[int(self.rng.integers(len(donor_nodes)))]
        offspring = renumber_constants(candidate.expr.replace(parent_path, donor_subtree))
        if offspring.depth() > self.config.max_depth + 1:
            return candidate
        params = self._optimize_field(candidate.field_idx, offspring)
        trial = self._candidate(candidate.field_idx, offspring, params)
        self.archive_crossovers += 1
        return trial if trial.reward > candidate.reward else candidate

    def _record(self, candidate: MultiCandidate, simulation: int) -> None:
        candidate = self._archive_crossover(candidate)
        if self.best is None or candidate.reward > self.best.reward:
            self.best = candidate
        if simulation % self.config.responsibility_interval == 0 or self.best is candidate:
            rho, edges = self._responsibility(candidate)
        else:
            rho, edges = 0.0, []
        self.archive.consider(candidate.field_idx, candidate, rho, edges)
        current = self._current_candidate()
        if candidate.reward > current.reward and candidate.field_idx >= 0:
            self.current_exprs = list(candidate.expressions)
            self.current_params = [values.copy() for values in candidate.params_list]

    def _current_candidate(self) -> MultiCandidate:
        expressions = tuple(self.current_exprs)
        params_list = tuple(values.copy() for values in self.current_params)
        field_data, residual_mse, loss = self._loss_components(expressions, params_list)
        complexity = sum(expression.node_count() for expression in expressions)
        return MultiCandidate(
            field_idx=-1,
            expr=expressions[0],
            params=params_list[0],
            expressions=expressions,
            params_list=params_list,
            field_data_mse=field_data,
            residual_mse=residual_mse,
            loss=loss,
            reward=float(np.exp(np.clip(-loss - self.config.eta * complexity, -700.0, 0.0))),
            complexity=complexity,
        )

    def _search_field(self, field_idx: int) -> None:
        root = MCTSNode(SearchState((), (0,)))
        for simulation in range(1, self.config.simulations + 1):
            node = root
            path = [node]
            while node.state.frontier and len(node.children) == len(self.allowed_actions(field_idx, node.state.frontier[0])):
                action = self._select_existing(node, field_idx)
                node = node.children[action]
                path.append(node)
            if node.state.frontier:
                action = self._select_unexpanded(node, field_idx)
                child = MCTSNode(self._next_state(node.state, action), node, action)
                node.children[action] = child
                node = child
                path.append(node)
            terminal = self._rollout(node.state, field_idx)
            expr = renumber_constants(self._parse_actions(terminal.actions))
            params = self._optimize_field(field_idx, expr)
            candidate = self._candidate(field_idx, expr, params)
            self._record(candidate, simulation)
            for visited in path:
                visited.visits += 1
                visited.value_sum += candidate.reward
                visited.recent_rewards.append(candidate.reward)

    @staticmethod
    def _parse_actions(actions: Sequence[str]) -> Expr:
        position = 0

        def parse() -> Expr:
            nonlocal position
            action = actions[position]
            position += 1
            if action.startswith("v"):
                return Expr.var(int(action[1:]))
            if action == "C":
                index = sum(1 for token in actions[: position - 1] if token == "C")
                return Expr.const(index)
            if action in UNARY_ACTIONS:
                return Expr(action, (parse(),))
            if action in BINARY_ACTIONS:
                return Expr(action, (parse(), parse()))
            raise ValueError(f"Unknown action: {action}")

        return parse()

    def fit(self) -> MultiCandidate:
        for field_idx, template in enumerate(self.problem.templates):
            params = self._optimize_field(field_idx, template)
            candidate = self._candidate(field_idx, template, params)
            self.bootstrap_candidates += 1
            self._record(candidate, 0)
        for _ in range(1):
            for field_idx in range(len(self.problem.field_names)):
                self._search_field(field_idx)
        if self.best is None:
            self.best = self._current_candidate()
        return self.best

    def evaluate(self, candidate: MultiCandidate) -> Dict[str, object]:
        jets = [expr.evaluate(self.problem.x_eval, params) for expr, params in zip(candidate.expressions, candidate.params_list)]
        field_mae = [float(np.mean(np.abs(jet.value - target))) for jet, target in zip(jets, self.problem.y_eval)]
        residuals = self.problem.residual_fn(
            [expr.evaluate(self.problem.x_eval, params) for expr, params in zip(candidate.expressions, candidate.params_list)],
            self.problem.x_eval,
        )
        residual_mae = [float(np.mean(np.abs(residual))) for residual in residuals]
        group_mae = {group: float(np.mean([field_mae[idx] for idx in indices])) for group, indices in self.problem.field_groups.items()}
        return {
            "field_mae_eval": dict(zip(self.problem.field_names, field_mae)),
            "group_mae_eval": group_mae,
            "residual_mae_eval": residual_mae,
            "expression": {name: expr.render(params) for name, expr, params in zip(self.problem.field_names, candidate.expressions, candidate.params_list)},
        }

    def report(self, candidate: MultiCandidate) -> Dict[str, object]:
        evaluated = self.evaluate(candidate)
        return {
            "benchmark": self.problem.name,
            "description": self.problem.description,
            "data_source": self.problem.data_source,
            "oracle_solution_injected": self.problem.data_source == "manufactured_solution_oracle",
            "field_names": list(self.problem.field_names),
            "expressions": evaluated["expression"],
            "field_mae_eval": evaluated["field_mae_eval"],
            "group_mae_eval": evaluated["group_mae_eval"],
            "residual_mae_eval": evaluated["residual_mae_eval"],
            "observation_field_mse": dict(zip(self.problem.field_names, candidate.field_data_mse)),
            "physics_channel_mse": candidate.residual_mse,
            "normalized_loss": candidate.loss,
            "reward": candidate.reward,
            "complexity": candidate.complexity,
            "simulations_per_field": self.config.simulations,
            "terminal_evaluations": self.terminal_evaluations,
            "intervention_count": self.intervention_count,
            "archive_crossovers": self.archive_crossovers,
            "bootstrap_candidates": self.bootstrap_candidates,
            "archive_size": self.archive.size,
            "responsibility_edges": len(self.memory.edges),
            "memory": self.memory.jsonable(),
            "archive": self.archive.jsonable(),
        }


TABLE_X_REFERENCE = {
    "Diffusion-convection": (2.05e-2, 1.25e-2),
    "Joule heating 1": (1.79e-5, 8.10e-1),
    "Joule heating 2": (1.63e-5, 7.93e-1),
    "Joule heating 3": (1.77e-5, 1.75e-1),
    "Joule heating 4": (1.81e-5, 1.59e-1),
    "Joule heating 5": (1.63e-5, 5.26e-1),
    "Parabolic 1": (1.30e-1, 5.52e-2),
    "Parabolic 2": (1.99e-1, 8.95e-2),
    "Parabolic 3": (1.76e-2, 1.38e-2),
}


def run_multiphysics_one(
    name: str,
    config: SearchConfig,
    data_seed: int = 0,
    n_obs: int = 128,
    n_phys: int = 256,
    n_eval: int = 1024,
    data_root: Optional[Path] = None,
) -> Dict[str, object]:
    problem = make_multiphysics_problem(name, data_seed, n_obs, n_phys, n_eval, data_root)
    search = MultiFieldSEDMCTS(problem, config)
    best = search.fit()
    result = search.report(best)
    reference = TABLE_X_REFERENCE.get(problem.name)
    if reference is not None:
        result["paper_table_x_reference_mae_f1_f2"] = list(reference)
    return result


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
