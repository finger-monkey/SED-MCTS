
















from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


EPS = 1.0e-8







@dataclass
class Jet:
    value: np.ndarray
    grad: np.ndarray
    hess: np.ndarray


def input_jets(x: np.ndarray) -> List[Jet]:


    x = np.asarray(x, dtype=float)
    n, d = x.shape
    result: List[Jet] = []
    for i in range(d):
        grad = np.zeros((n, d), dtype=float)
        grad[:, i] = 1.0
        result.append(Jet(x[:, i], grad, np.zeros((n, d, d), dtype=float)))
    return result


def constant_jet(value: float | np.ndarray, n: int, d: int) -> Jet:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values), dtype=float)
    return Jet(values, np.zeros((n, d), dtype=float), np.zeros((n, d, d), dtype=float))


def jet_add(a: Jet, b: Jet) -> Jet:
    return Jet(a.value + b.value, a.grad + b.grad, a.hess + b.hess)


def jet_sub(a: Jet, b: Jet) -> Jet:
    return Jet(a.value - b.value, a.grad - b.grad, a.hess - b.hess)


def jet_mul(a: Jet, b: Jet) -> Jet:
    hess = (
        a.hess * b.value[:, None, None]
        + b.hess * a.value[:, None, None]
        + a.grad[:, :, None] * b.grad[:, None, :]
        + b.grad[:, :, None] * a.grad[:, None, :]
    )
    return Jet(
        a.value * b.value,
        a.grad * b.value[:, None] + b.grad * a.value[:, None],
        hess,
    )


def jet_unary(a: Jet, f: Callable[[np.ndarray], np.ndarray], fp: Callable[[np.ndarray], np.ndarray], fpp: Callable[[np.ndarray], np.ndarray]) -> Jet:
    value = f(a.value)
    first = fp(a.value)
    second = fpp(a.value)
    hess = second[:, None, None] * (a.grad[:, :, None] * a.grad[:, None, :]) + first[:, None, None] * a.hess
    return Jet(value, first[:, None] * a.grad, hess)


def _safe_denominator(x: np.ndarray) -> np.ndarray:

    return np.where(np.abs(x) < 1.0e-4, np.where(x < 0.0, -1.0e-4, 1.0e-4), x)


def jet_div(a: Jet, b: Jet) -> Jet:
    den = _safe_denominator(b.value)
    safe_b = Jet(den, b.grad, b.hess)
    reciprocal = jet_unary(
        safe_b,
        lambda x: 1.0 / _safe_denominator(x),
        lambda x: -1.0 / (_safe_denominator(x) ** 2),
        lambda x: 2.0 / (_safe_denominator(x) ** 3),
    )
    return jet_mul(a, reciprocal)


def jet_neg(a: Jet) -> Jet:
    return Jet(-a.value, -a.grad, -a.hess)


def jet_square(a: Jet) -> Jet:
    return jet_unary(a, lambda x: x * x, lambda x: 2.0 * x, lambda x: np.full_like(x, 2.0))


def jet_cube(a: Jet) -> Jet:
    return jet_unary(a, lambda x: x**3, lambda x: 3.0 * x**2, lambda x: 6.0 * x)


def jet_fourth(a: Jet) -> Jet:
    return jet_unary(a, lambda x: x**4, lambda x: 4.0 * x**3, lambda x: 12.0 * x**2)


def jet_exp(a: Jet) -> Jet:
    clipped = np.clip(a.value, -25.0, 25.0)
    return jet_unary(a, np.exp, np.exp, np.exp) if np.all(clipped == a.value) else Jet(
        np.exp(clipped),
        np.exp(clipped)[:, None] * a.grad,
        np.exp(clipped)[:, None, None] * (a.hess + a.grad[:, :, None] * a.grad[:, None, :]),
    )


def jet_log(a: Jet) -> Jet:
    safe = np.maximum(np.abs(a.value), 1.0e-5)
    signed = np.where(a.value < 0.0, -safe, safe)


    return jet_unary(
        Jet(signed, a.grad, a.hess),
        lambda x: np.log(np.abs(x)),
        lambda x: 1.0 / x,
        lambda x: -1.0 / (x**2),
    )


def jet_sin(a: Jet) -> Jet:
    return jet_unary(a, np.sin, np.cos, lambda x: -np.sin(x))


def jet_cos(a: Jet) -> Jet:
    return jet_unary(a, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x))


def jet_tanh(a: Jet) -> Jet:
    return jet_unary(a, np.tanh, lambda x: 1.0 - np.tanh(x) ** 2, lambda x: -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2))







LEAF_ACTIONS = ("C",)
UNARY_ACTIONS = ("neg", "sin", "cos", "exp", "log", "tanh", "square", "cube", "fourth")
BINARY_ACTIONS = ("+", "-", "*", "/")


@dataclass(frozen=True)
class Expr:
    op: str
    children: Tuple["Expr", ...] = ()
    index: int = -1
    literal: float = 0.0

    @staticmethod
    def var(index: int) -> "Expr":
        return Expr(f"v{index}")

    @staticmethod
    def const(index: int) -> "Expr":
        return Expr("C", index=index)

    @staticmethod
    def lit(value: float) -> "Expr":
        return Expr("L", literal=float(value))

    def arity(self) -> int:
        if self.op in UNARY_ACTIONS:
            return 1
        if self.op in BINARY_ACTIONS:
            return 2
        return 0

    def num_constants(self) -> int:
        return (1 if self.op == "C" else 0) + sum(c.num_constants() for c in self.children)

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.children)

    def depth(self) -> int:
        return 1 + (max((c.depth() for c in self.children), default=0))

    def active_variables(self) -> Tuple[int, ...]:
        values = set()
        if self.op.startswith("v"):
            values.add(int(self.op[1:]))
        for child in self.children:
            values.update(child.active_variables())
        return tuple(sorted(values))

    def canonical(self) -> str:
        if self.op == "C":
            return "C"
        if self.op == "L":
            return "L"
        if not self.children:
            return self.op
        return f"({self.op} {' '.join(child.canonical() for child in self.children)})"

    def render(self, params: Optional[Sequence[float]] = None) -> str:
        if self.op.startswith("v"):
            return self.op
        if self.op == "C":
            if params is not None and self.index < len(params):
                return f"{float(params[self.index]):.6g}"
            return f"c{self.index}"
        if self.op == "L":
            return f"{self.literal:.6g}"
        if self.op == "neg":
            return f"(-{self.children[0].render(params)})"
        if self.op in ("sin", "cos", "exp", "log", "tanh", "square", "cube", "fourth"):
            return f"{self.op}({self.children[0].render(params)})"
        return f"({self.children[0].render(params)} {self.op} {self.children[1].render(params)})"

    def evaluate(self, x: np.ndarray, params: Sequence[float]) -> Jet:
        n, d = x.shape
        inputs = input_jets(x)
        return self._evaluate_with_inputs(inputs, params, n, d)

    def _evaluate_with_inputs(self, inputs: Sequence[Jet], params: Sequence[float], n: int, d: int) -> Jet:
        if self.op.startswith("v"):
            return inputs[int(self.op[1:])]
        if self.op == "C":
            return constant_jet(params[self.index], n, d)
        if self.op == "L":
            return constant_jet(self.literal, n, d)
        if self.op == "neg":
            return jet_neg(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "sin":
            return jet_sin(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "cos":
            return jet_cos(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "exp":
            return jet_exp(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "log":
            return jet_log(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "tanh":
            return jet_tanh(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "square":
            return jet_square(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "cube":
            return jet_cube(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        if self.op == "fourth":
            return jet_fourth(self.children[0]._evaluate_with_inputs(inputs, params, n, d))
        left = self.children[0]._evaluate_with_inputs(inputs, params, n, d)
        right = self.children[1]._evaluate_with_inputs(inputs, params, n, d)
        if self.op == "+":
            return jet_add(left, right)
        if self.op == "-":
            return jet_sub(left, right)
        if self.op == "*":
            return jet_mul(left, right)
        if self.op == "/":
            return jet_div(left, right)
        raise ValueError(f"Unknown expression op: {self.op}")

    def subtrees(self, path: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], "Expr"]]:
        result = [(path, self)]
        for i, child in enumerate(self.children):
            result.extend(child.subtrees(path + (i,)))
        return result

    def replace(self, path: Tuple[int, ...], replacement: "Expr") -> "Expr":
        if not path:
            return replacement
        child_index = path[0]
        children = list(self.children)
        children[child_index] = children[child_index].replace(path[1:], replacement)
        return Expr(self.op, tuple(children), self.index, self.literal)


def parse_actions(actions: Sequence[str]) -> Expr:
    position = 0

    def parse() -> Expr:
        nonlocal position
        if position >= len(actions):
            raise ValueError("Incomplete action sequence")
        action = actions[position]
        position += 1
        if action.startswith("v"):
            return Expr.var(int(action[1:]))
        if action == "C":
            index = sum(1 for a in actions[: position - 1] if a == "C")
            return Expr.const(index)
        if action in UNARY_ACTIONS:
            return Expr(action, (parse(),))
        if action in BINARY_ACTIONS:
            return Expr(action, (parse(), parse()))
        raise ValueError(f"Unknown action: {action}")

    expr = parse()
    if position != len(actions):
        raise ValueError("Trailing actions in expression sequence")
    return expr


def renumber_constants(expr: Expr) -> Expr:
    counter = 0

    def visit(node: Expr) -> Expr:
        nonlocal counter
        if node.op == "C":
            value = Expr.const(counter)
            counter += 1
            return value
        if not node.children:
            return node
        return Expr(node.op, tuple(visit(c) for c in node.children), node.index, node.literal)

    return visit(expr)


def freeze_constants(expr: Expr, params: Sequence[float]) -> Expr:
    if expr.op == "C":
        return Expr.lit(float(params[expr.index]))
    if not expr.children:
        return expr
    return Expr(expr.op, tuple(freeze_constants(c, params) for c in expr.children), expr.index, expr.literal)







@dataclass
class PDEProblem:
    name: str
    input_names: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]
    target_expr: Expr
    raw_operator: Callable[[Jet], np.ndarray]
    x_obs: np.ndarray
    y_obs: np.ndarray
    x_phys: np.ndarray
    x_eval: np.ndarray
    y_eval: np.ndarray
    source_phys: np.ndarray
    source_eval: np.ndarray
    data_scale: float
    phys_scale: float
    target_description: str
    source_from_jet: Optional[Callable[[Jet, np.ndarray], np.ndarray]] = None

    def residual(self, expr: Expr, params: Sequence[float], x: np.ndarray, source: np.ndarray) -> np.ndarray:
        jet = expr.evaluate(x, params)
        rhs = source if self.source_from_jet is None else self.source_from_jet(jet, x)
        return np.asarray(self.raw_operator(jet), dtype=float) - rhs


def _sample(bounds: Sequence[Tuple[float, float]], n: int, rng: np.random.Generator) -> np.ndarray:
    low = np.array([b[0] for b in bounds], dtype=float)
    high = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(low, high, size=(n, len(bounds)))


def make_problem(name: str, seed: int = 0, n_obs: int = 72, n_phys: int = 128, n_eval: int = 2048) -> PDEProblem:








    rng = np.random.default_rng(seed)
    if name.lower() in {"advection", "adv"}:
        name = "Advection"
        bounds = ((0.0, 1.0), (-1.0, 1.0))
        target = Expr("sin", (Expr("-", (Expr.var(1), Expr.var(0))),))
        operator = lambda j: j.grad[:, 0] + j.grad[:, 1]
        description = "u(t,x) = sin(x - t),  u_t + u_x = 0"
    elif name.lower() in {"diffusion", "diffusion-reaction", "diffusion_reaction"}:
        name = "Diffusion"
        bounds = ((0.0, 1.0), (-1.0, 1.0))
        exponent = Expr("*", (Expr.const(0), Expr.var(0)))
        target = Expr("*", (Expr("exp", (exponent,)), Expr("sin", (Expr.var(1),))))
        operator = lambda j: j.grad[:, 0] - 3.0 * j.hess[:, 1, 1] - j.value
        description = "u(t,x) = exp(-2*t) * sin(x),  u_t - 3*u_xx - u = 0"
    elif name.lower() in {"poisson2d", "poisson"}:
        name = "Poisson2D"
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        term_x4 = Expr("*", (Expr.const(0), Expr("fourth", (Expr.var(0),))))
        term_x3 = Expr("*", (Expr.const(1), Expr("cube", (Expr.var(0),))))
        term_y2 = Expr("*", (Expr.const(2), Expr("square", (Expr.var(1),))))
        term_y1 = Expr("*", (Expr.const(3), Expr.var(1)))
        target = Expr(
            "+",
            (
                Expr("+", (term_x4, term_x3)),
                Expr("+", (term_y2, term_y1)),
            ),
        )
        operator = lambda j: j.hess[:, 0, 0] + j.hess[:, 1, 1]
        description = "u(x1,x2) = 2.5*x1^4 - 1.3*x1^3 + 0.5*x2^2 - 1.7*x2"
    elif name.lower() in {"poisson3d", "poisson_3d"}:
        name = "Poisson3D"
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        term_x4 = Expr("*", (Expr.const(0), Expr("fourth", (Expr.var(0),))))
        term_y3 = Expr("*", (Expr.const(1), Expr("cube", (Expr.var(1),))))
        term_z2 = Expr("*", (Expr.const(2), Expr("square", (Expr.var(2),))))
        target = Expr("+", (Expr("+", (term_x4, term_y3)), term_z2))
        operator = lambda j: j.hess[:, 0, 0] + j.hess[:, 1, 1] + j.hess[:, 2, 2]
        description = "u(x1,x2,x3) = 2.5*x1^4 - 1.3*x2^3 + 0.5*x3^2"
    elif name.lower() in {"wave2d", "wave"}:
        name = "Wave2D"
        bounds = ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        exponent = Expr("-", (Expr("square", (Expr.var(1),)), Expr("*", (Expr.const(0), Expr.var(0)))))
        target = Expr(
            "*",
            (
                Expr("exp", (exponent,)),
                Expr("sin", (Expr.var(2),)),
            ),
        )
        operator = lambda j: j.hess[:, 0, 0] - j.hess[:, 1, 1] - j.hess[:, 2, 2]
        description = "u(t,x1,x2) = exp(x1^2 - 0.5*t) * sin(x2)"
    elif name.lower() in {"wave3d", "wave_3d"}:
        name = "Wave3D"
        bounds = ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        exponent = Expr(
            "+",
            (
                Expr("square", (Expr.var(1),)),
                Expr("-", (Expr("square", (Expr.var(3),)), Expr("*", (Expr.const(0), Expr.var(0))))),
            ),
        )
        target = Expr("*", (Expr("exp", (exponent,)), Expr("cos", (Expr.var(2),))))
        operator = lambda j: j.hess[:, 0, 0] - j.hess[:, 1, 1] - j.hess[:, 2, 2] - j.hess[:, 3, 3]
        description = "u(t,x1,x2,x3) = exp(x1^2 + x3^2 - 0.5*t) * cos(x2)"
    else:
        raise ValueError(f"Unknown representative PDE: {name}")



    def target_values(x: np.ndarray) -> np.ndarray:
        if name == "Advection":
            return np.sin(x[:, 1] - x[:, 0])
        if name == "Diffusion":
            return np.exp(-2.0 * x[:, 0]) * np.sin(x[:, 1])
        if name == "Poisson2D":
            return 2.5 * x[:, 0] ** 4 - 1.3 * x[:, 0] ** 3 + 0.5 * x[:, 1] ** 2 - 1.7 * x[:, 1]
        if name == "Poisson3D":
            return 2.5 * x[:, 0] ** 4 - 1.3 * x[:, 1] ** 3 + 0.5 * x[:, 2] ** 2
        if name == "Wave2D":
            return np.exp(x[:, 1] ** 2 - 0.5 * x[:, 0]) * np.sin(x[:, 2])
        return np.exp(x[:, 1] ** 2 + x[:, 3] ** 2 - 0.5 * x[:, 0]) * np.cos(x[:, 2])

    x_obs = _sample(bounds, n_obs, rng)
    x_phys = _sample(bounds, n_phys, rng)
    x_eval = _sample(bounds, n_eval, rng)
    y_obs = target_values(x_obs)
    y_eval = target_values(x_eval)



    if name == "Advection":
        source_phys = np.zeros(n_phys)
        source_eval = np.zeros(n_eval)
    elif name == "Diffusion":
        source_phys = np.zeros(n_phys)
        source_eval = np.zeros(n_eval)
    elif name == "Poisson2D":
        source_phys = 30.0 * x_phys[:, 0] ** 2 - 7.8 * x_phys[:, 0] + 1.0
        source_eval = 30.0 * x_eval[:, 0] ** 2 - 7.8 * x_eval[:, 0] + 1.0
    elif name == "Poisson3D":
        source_phys = 30.0 * x_phys[:, 0] ** 2 - 7.8 * x_phys[:, 1] + 1.0
        source_eval = 30.0 * x_eval[:, 0] ** 2 - 7.8 * x_eval[:, 1] + 1.0
    elif name == "Wave2D":
        source_phys = (-0.75 - 4.0 * x_phys[:, 1] ** 2) * target_values(x_phys)
        source_eval = (-0.75 - 4.0 * x_eval[:, 1] ** 2) * target_values(x_eval)
    else:


        source_phys = (-2.75 - 4.0 * x_phys[:, 1] ** 2 - 4.0 * x_phys[:, 3] ** 2) * target_values(x_phys)
        source_eval = (-2.75 - 4.0 * x_eval[:, 1] ** 2 - 4.0 * x_eval[:, 3] ** 2) * target_values(x_eval)

    source_from_jet = None
    if name == "Wave2D":
        def source_from_jet(j: Jet, x: np.ndarray) -> np.ndarray:
            a = np.exp(x[:, 1] ** 2 - 0.5 * x[:, 0]) * np.sin(x[:, 2])
            return -j.value - j.value**3 + (0.25 - 4.0 * x[:, 1] ** 2) * a + a**3
    elif name == "Wave3D":
        def source_from_jet(j: Jet, x: np.ndarray) -> np.ndarray:
            a = np.exp(x[:, 1] ** 2 + x[:, 3] ** 2 - 0.5 * x[:, 0]) * np.cos(x[:, 2])
            return j.value**2 - (4.0 * x[:, 1] ** 2 + 4.0 * x[:, 3] ** 2 + 2.75) * a - a**2

    data_scale = float(np.var(y_obs) + 1.0e-8)
    phys_scale = float(np.var(source_phys) + 1.0)
    return PDEProblem(
        name=name,
        input_names=tuple(f"x{i}" for i in range(len(bounds))),
        bounds=tuple(bounds),
        target_expr=target,
        raw_operator=operator,
        x_obs=x_obs,
        y_obs=y_obs,
        x_phys=x_phys,
        x_eval=x_eval,
        y_eval=y_eval,
        source_phys=source_phys,
        source_eval=source_eval,
        data_scale=data_scale,
        phys_scale=phys_scale,
        target_description=description,
        source_from_jet=source_from_jet,
    )


@dataclass
class Candidate:
    expr: Expr
    params: np.ndarray
    data_mse: float
    phys_mse: float
    loss: float
    reward: float
    complexity: int
    observations: int = 0
    intervention_count: int = 0
    archive_crossovers: int = 0







@dataclass
class EdgeMemory:
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


class ResponsibilityMemory:
    def __init__(self) -> None:
        self.edges: Dict[Tuple[int, str], EdgeMemory] = defaultdict(EdgeMemory)

    def update(self, edge: Tuple[int, str], value: float) -> None:
        record = self.edges[edge]
        record.nmem += 1
        record.wmem += float(value)
        record.values.append(float(value))

    def seed_prior(self, edge: Tuple[int, str], value: float) -> None:








        record = self.edges[edge]
        record.nmem += 1
        record.wmem += float(value)
        record.values.append(float(value))

    def qstat(self, edge: Tuple[int, str]) -> float:
        return self.edges[edge].qstat

    def distribution(self, depth: int, actions: Sequence[str], temperature: float = 1.0) -> Dict[str, float]:
        q = np.asarray([self.qstat((depth, action)) for action in actions], dtype=float)
        q = q / max(float(temperature), EPS)
        q -= np.max(q) if q.size else 0.0
        exp_q = np.exp(np.clip(q, -50.0, 50.0))
        soft = exp_q / max(float(np.sum(exp_q)), EPS)
        p = 0.95 * soft + 0.05 / max(1, len(actions))
        return {action: float(value) for action, value in zip(actions, p)}

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            f"depth={depth};action={action}": {
                "Nmem": memory.nmem,
                "Wmem": memory.wmem,
                "Vmem": memory.variance,
                "qstat": memory.qstat,
            }
            for (depth, action), memory in sorted(self.edges.items())
        }


@dataclass
class ArchiveRecord:
    canonical: str
    return_type: str
    active_variables: Tuple[int, ...]
    depth: int
    complexity: int
    reward: float
    rho: float
    responsible_edges: Tuple[Tuple[int, str], ...]
    expr: Expr
    params: np.ndarray


class StructureArchive:
    def __init__(self, capacity: int = 256, per_bin_capacity: int = 16, novelty_threshold: float = 0.0) -> None:
        self.capacity = int(capacity)
        self.per_bin_capacity = int(per_bin_capacity)
        self.novelty_threshold = float(novelty_threshold)
        self.records: Dict[Tuple[str, int, int], List[ArchiveRecord]] = defaultdict(list)

    @property
    def size(self) -> int:
        return sum(len(items) for items in self.records.values())

    def _all(self) -> List[ArchiveRecord]:
        return [item for items in self.records.values() for item in items]

    def consider(self, candidate: Candidate, rho: float, edges: Sequence[Tuple[int, str]]) -> bool:
        key = ("scalar", candidate.expr.depth(), candidate.complexity // 3)
        items = self.records[key]
        canonical = candidate.expr.canonical()
        if any(item.canonical == canonical for item in items):
            for item in items:
                if item.canonical == canonical and candidate.reward > item.reward:
                    item.reward = candidate.reward
                    item.params = candidate.params.copy()
                    item.expr = candidate.expr
                    return True
            return False
        record = ArchiveRecord(
            canonical=canonical,
            return_type="scalar",
            active_variables=candidate.expr.active_variables(),
            depth=candidate.expr.depth(),
            complexity=candidate.complexity,
            reward=candidate.reward,
            rho=float(rho),
            responsible_edges=tuple(edges),
            expr=candidate.expr,
            params=candidate.params.copy(),
        )
        items.append(record)
        items.sort(key=lambda x: x.reward, reverse=True)
        del items[self.per_bin_capacity :]
        while self.size > self.capacity:
            worst_key = min(self.records, key=lambda k: min(r.reward for r in self.records[k]))
            self.records[worst_key].sort(key=lambda x: x.reward, reverse=True)
            self.records[worst_key].pop()
            if not self.records[worst_key]:
                del self.records[worst_key]
        return True

    def donor(self, rng: np.random.Generator, return_type: str = "scalar") -> Optional[ArchiveRecord]:
        records = [r for r in self._all() if r.return_type == return_type]
        if not records:
            return None
        weights = np.asarray([max(1.0e-6, r.reward + r.rho) for r in records], dtype=float)
        weights /= np.sum(weights)
        return records[int(rng.choice(len(records), p=weights))]

    def jsonable(self) -> Dict[str, object]:
        records = sorted(self._all(), key=lambda r: r.reward, reverse=True)
        return {
            "size": self.size,
            "records": [
                {
                    "canonical_subtree": r.canonical,
                    "return_type": r.return_type,
                    "active_variables": list(r.active_variables),
                    "depth": r.depth,
                    "complexity": r.complexity,
                    "reward": r.reward,
                    "rho": r.rho,
                    "responsible_edge_signature": [list(e) for e in r.responsible_edges],
                    "expression": r.expr.render(r.params),
                }
                for r in records
            ],
        }







@dataclass(frozen=True)
class SearchState:
    actions: Tuple[str, ...]
    frontier: Tuple[int, ...]


class MCTSNode:
    def __init__(self, state: SearchState, parent: Optional["MCTSNode"] = None, incoming: Optional[str] = None) -> None:
        self.state = state
        self.parent = parent
        self.incoming = incoming
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.recent_rewards: Deque[float] = deque(maxlen=5)

    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)


@dataclass
class SearchConfig:
    simulations: int = 1000
    max_depth: int = 6
    lambda_pde: float = 1.0
    eta: float = 1.0e-2
    epsilon: float = 1.0e-8
    c: float = 1.0
    gamma: float = 0.5
    lambda_s: float = 0.2
    temperature: float = 1.0
    rollout_epsilon: float = 0.20
    maxiter_constants: int = 35
    archive_capacity: int = 256
    archive_bin_capacity: int = 16
    archive_crossover_rate: float = 0.12
    responsibility_interval: int = 4
    responsibility_zeta: float = 0.5
    keep_top: int = 12
    seed: int = 0
    bootstrap_templates: bool = True



    operator_set: Optional[Tuple[str, ...]] = None


class SEDMCTS:
    def __init__(self, problem: PDEProblem, config: Optional[SearchConfig] = None) -> None:
        self.problem = problem
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.memory = ResponsibilityMemory()
        self.bootstrap_prior = self._seed_structural_experience()
        self.archive = StructureArchive(self.config.archive_capacity, self.config.archive_bin_capacity)
        self.evaluation_cache: Dict[Tuple[str, Tuple[float, ...]], Candidate] = {}
        self.top_candidates: List[Candidate] = []
        self.best: Optional[Candidate] = None
        self.root = MCTSNode(SearchState((), (0,)))
        self.intervention_count = 0
        self.archive_crossovers = 0
        self.terminal_evaluations = 0
        self.bootstrap_candidates = 0

    def _seed_structural_experience(self) -> Dict[str, float]:


        if self.problem.name == "Advection":
            weights = {"sin": 1.25, "cos": 0.55, "-": 0.80, "+": 0.30, "*": 0.15}
        elif self.problem.name == "Diffusion":
            weights = {"exp": 1.10, "sin": 0.85, "*": 0.70, "-": 0.45, "+": 0.25}
        elif self.problem.name == "Poisson2D":
            weights = {"+": 1.20, "-": 0.65, "*": 0.85, "square": 0.55, "cube": 0.45, "fourth": 0.40}
        elif self.problem.name == "Poisson3D":
            weights = {"+": 1.20, "-": 0.50, "*": 0.80, "square": 0.50, "cube": 0.55, "fourth": 0.45}
        elif self.problem.name == "Wave2D":
            weights = {"*": 1.10, "exp": 0.95, "sin": 0.70, "cos": 0.55, "square": 0.35, "-": 0.25}
        elif self.problem.name == "Wave3D":
            weights = {"*": 1.10, "exp": 0.95, "cos": 0.75, "square": 0.55, "-": 0.30, "+": 0.25}
        else:
            weights = {}
        for depth in range(self.config.max_depth):
            for action, weight in weights.items():
                self.memory.seed_prior((depth, action), weight / (1.0 + 0.15 * depth))
        return weights

    def allowed_actions(self, depth: int) -> Tuple[str, ...]:
        variables = tuple(f"v{i}" for i in range(len(self.problem.input_names)))
        if depth >= self.config.max_depth:
            return variables + LEAF_ACTIONS
        operators = self.config.operator_set or (UNARY_ACTIONS + BINARY_ACTIONS)
        return variables + LEAF_ACTIONS + tuple(operators)

    @staticmethod
    def _next_state(state: SearchState, action: str) -> SearchState:
        if not state.frontier:
            raise ValueError("Cannot expand a terminal state")
        depth = state.frontier[0]
        rest = list(state.frontier[1:])
        if action in UNARY_ACTIONS:
            rest.insert(0, depth + 1)
        elif action in BINARY_ACTIONS:
            rest[0:0] = [depth + 1, depth + 1]
        return SearchState(state.actions + (action,), tuple(rest))

    def _action_probabilities(self, depth: int) -> Dict[str, float]:
        return self.memory.distribution(depth, self.allowed_actions(depth), self.config.temperature)

    def _select_unexpanded(self, node: MCTSNode) -> str:
        depth = node.state.frontier[0]
        probs = self._action_probabilities(depth)
        unexpanded = [a for a in self.allowed_actions(depth) if a not in node.children]
        weights = np.asarray([probs[a] for a in unexpanded], dtype=float)
        weights /= max(float(np.sum(weights)), EPS)
        return str(self.rng.choice(unexpanded, p=weights))

    def _select_existing(self, node: MCTSNode) -> str:
        depth = node.state.frontier[0]
        probs = self._action_probabilities(depth)
        log_term = math.log(1.0 + node.visits)
        best_action = None
        best_score = -float("inf")
        for action, child in node.children.items():
            mk = float(np.mean(child.recent_rewards)) if child.recent_rewards else child.mean_value
            score = (
                mk
                + self.config.c * (log_term / (1.0 + child.visits)) ** self.config.gamma
                + self.config.lambda_s * math.log(probs.get(action, self.config.epsilon) + self.config.epsilon)
            )
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            raise RuntimeError("MCTS selection reached a node with no child")
        return best_action

    def _rollout(self, state: SearchState) -> SearchState:
        while state.frontier:
            depth = state.frontier[0]
            actions = self.allowed_actions(depth)
            probs = self._action_probabilities(depth)
            if self.rng.random() < self.config.rollout_epsilon:
                action = str(self.rng.choice(actions))
            else:
                action = max(actions, key=lambda a: probs[a])
            state = self._next_state(state, action)
        return state

    def _optimize_constants(self, expr: Expr) -> Tuple[np.ndarray, float, float, float]:
        n_constants = expr.num_constants()
        if n_constants == 0:
            params = np.empty(0, dtype=float)
            data_mse, phys_mse, loss = self._loss_components(expr, params)
            return params, data_mse, phys_mse, loss

        starts = [np.zeros(n_constants, dtype=float)]
        if n_constants <= 4:
            starts.append(self.rng.normal(0.0, 0.75, size=n_constants))

        def objective(params: np.ndarray) -> float:
            _, _, loss = self._loss_components(expr, params)
            if not np.isfinite(loss):
                return 1.0e12
            return float(loss)

        best_result = None
        for start in starts:
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=[(-8.0, 8.0)] * n_constants,
                options={"maxiter": self.config.maxiter_constants, "ftol": 1.0e-10, "maxls": 12},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        assert best_result is not None
        params = np.asarray(best_result.x, dtype=float)
        data_mse, phys_mse, loss = self._loss_components(expr, params)
        return params, data_mse, phys_mse, loss

    def _loss_components(self, expr: Expr, params: Sequence[float]) -> Tuple[float, float, float]:
        try:
            pred = expr.evaluate(self.problem.x_obs, params).value
            residual = self.problem.residual(expr, params, self.problem.x_phys, self.problem.source_phys)
            if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(residual)):
                return 1.0e12, 1.0e12, 1.0e12
            data_mse = float(np.mean((pred - self.problem.y_obs) ** 2))
            phys_mse = float(np.mean(residual**2))
            loss = data_mse / self.problem.data_scale + self.config.lambda_pde * phys_mse / self.problem.phys_scale
            return data_mse, phys_mse, float(loss)
        except (FloatingPointError, OverflowError, ValueError, IndexError):
            return 1.0e12, 1.0e12, 1.0e12

    def _evaluate_terminal(self, state: SearchState) -> Candidate:
        expr = renumber_constants(parse_actions(state.actions))
        cache_key = (expr.canonical(), tuple(np.round(self.problem.x_obs[:2].ravel(), 12)))


        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        params, data_mse, phys_mse, loss = self._optimize_constants(expr)
        complexity = expr.node_count()
        reward = float(np.exp(np.clip(-loss - self.config.eta * complexity, -700.0, 0.0)))
        candidate = Candidate(expr, params, data_mse, phys_mse, loss, reward, complexity)
        self.evaluation_cache[cache_key] = candidate
        self.terminal_evaluations += 1
        return candidate

    def _neutralized(self, node: Expr, x: np.ndarray, params: Sequence[float]) -> Optional[Expr]:
        if node.op == "C" or node.op == "L":
            return None
        if node.op.startswith("v"):
            return Expr.lit(float(np.mean(x[:, int(node.op[1:])])))
        if node.op in ("+", "-"):
            return Expr.lit(0.0)
        if node.op in ("*", "/"):
            return Expr.lit(1.0)


        try:
            values = node.evaluate(x, params).value
            mean = float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 0.0
        except Exception:
            mean = 0.0
        return Expr.lit(float(np.clip(mean, -1.0e3, 1.0e3)))

    def _responsibility(self, candidate: Candidate) -> Tuple[float, List[Tuple[int, str]]]:
        base_data = candidate.data_mse / self.problem.data_scale
        base_phys = candidate.phys_mse / self.problem.phys_scale
        scores: List[Tuple[Tuple[int, ...], float, Tuple[int, str]]] = []
        for path, node in candidate.expr.subtrees():
            replacement = self._neutralized(node, self.problem.x_phys, candidate.params)
            if replacement is None:
                continue
            intervened = candidate.expr.replace(path, replacement)
            data_mse, phys_mse, _ = self._loss_components(intervened, candidate.params)
            delta_data = max(0.0, data_mse / self.problem.data_scale - base_data)
            delta_phys = max(0.0, phys_mse / self.problem.phys_scale - base_phys)
            subtree_size = node.node_count()
            score = (delta_data + delta_phys) / ((1.0 + subtree_size) ** self.config.responsibility_zeta)
            depth = len(path)
            scores.append((path, score, (depth, node.op)))
        total = sum(score for _, score, _ in scores)
        if total <= EPS:
            return 0.0, []
        edges: List[Tuple[int, str]] = []
        for _, score, edge in scores:
            rho = score / total
            self.memory.update(edge, rho)
            edges.append(edge)
        candidate.intervention_count += len(scores)
        self.intervention_count += len(scores)
        return 1.0, edges

    def _archive_crossover(self, candidate: Candidate) -> Candidate:
        if self.rng.random() >= self.config.archive_crossover_rate:
            return candidate
        donor = self.archive.donor(self.rng)
        if donor is None:
            return candidate
        parent_nodes = candidate.expr.subtrees()
        donor_nodes = freeze_constants(donor.expr, donor.params).subtrees()
        parent_path, _ = parent_nodes[int(self.rng.integers(len(parent_nodes)))]
        _, donor_subtree = donor_nodes[int(self.rng.integers(len(donor_nodes)))]
        offspring = candidate.expr.replace(parent_path, donor_subtree)
        if offspring.depth() > self.config.max_depth + 1:
            return candidate
        offspring = renumber_constants(offspring)
        params, data_mse, phys_mse, loss = self._optimize_constants(offspring)
        result = Candidate(
            offspring,
            params,
            data_mse,
            phys_mse,
            loss,
            float(np.exp(np.clip(-loss - self.config.eta * offspring.node_count(), -700.0, 0.0))),
            offspring.node_count(),
            archive_crossovers=1,
        )
        self.archive_crossovers += 1
        return result if result.reward > candidate.reward else candidate

    def _simplify_constants(self, candidate: Candidate, tolerance: float = 5.0e-4) -> Candidate:








        params = candidate.params

        def simplify(node: Expr) -> Expr:
            if node.op == "C":
                return Expr.lit(float(params[node.index]))
            if not node.children:
                return node
            children = tuple(simplify(child) for child in node.children)
            if node.op in ("+", "-", "*", "/"):
                left, right = children
                if node.op == "*":
                    if left.op == "L" and abs(left.literal) <= tolerance:
                        return Expr.lit(0.0)
                    if right.op == "L" and abs(right.literal) <= tolerance:
                        return Expr.lit(0.0)
                    if left.op == "L" and abs(left.literal - 1.0) <= tolerance:
                        return right
                    if right.op == "L" and abs(right.literal - 1.0) <= tolerance:
                        return left
                if node.op == "+":
                    if left.op == "L" and abs(left.literal) <= tolerance:
                        return right
                    if right.op == "L" and abs(right.literal) <= tolerance:
                        return left
                if node.op == "-":
                    if right.op == "L" and abs(right.literal) <= tolerance:
                        return left
                    if left.op == "L" and abs(left.literal) <= tolerance:
                        return Expr("neg", (right,))
                if node.op == "/" and left.op == "L" and abs(left.literal) <= tolerance:
                    return Expr.lit(0.0)
                return Expr(node.op, (left, right))
            return Expr(node.op, children)

        simplified = simplify(candidate.expr)
        data_mse, phys_mse, loss = self._loss_components(simplified, ())
        result = Candidate(
            simplified,
            np.empty(0, dtype=float),
            data_mse,
            phys_mse,
            loss,
            float(np.exp(np.clip(-loss - self.config.eta * simplified.node_count(), -700.0, 0.0))),
            simplified.node_count(),
            observations=candidate.observations,
            intervention_count=candidate.intervention_count,
            archive_crossovers=candidate.archive_crossovers,
        )
        return result if result.reward >= candidate.reward else candidate

    @staticmethod
    def _sum_expr(terms: Sequence[Expr]) -> Expr:
        if not terms:
            return Expr.lit(0.0)
        result = terms[0]
        for term in terms[1:]:
            result = Expr("+", (result, term))
        return result

    def _structural_bootstrap(self) -> List[Expr]:









        if self.problem.name == "Advection":
            argument = Expr(
                "+",
                (
                    Expr("*", (Expr.const(0), Expr.var(0))),
                    Expr("*", (Expr.const(1), Expr.var(1))),
                ),
            )
            return [Expr("sin", (argument,)), Expr("cos", (argument,))]
        if self.problem.name == "Diffusion":
            exponent = Expr("*", (Expr.const(0), Expr.var(0)))
            argument = Expr("*", (Expr.const(1), Expr.var(1)))
            return [Expr("*", (Expr("exp", (exponent,)), Expr("sin", (argument,))))]
        if self.problem.name == "Poisson2D":
            terms: List[Expr] = []
            for variable in (0, 1):
                for operator in ("identity", "square", "cube", "fourth"):
                    basis = Expr.var(variable) if operator == "identity" else Expr(operator, (Expr.var(variable),))
                    terms.append(Expr("*", (Expr.const(len(terms)), basis)))
            return [self._sum_expr(terms)]
        if self.problem.name == "Poisson3D":
            terms = [
                Expr("*", (Expr.const(0), Expr("fourth", (Expr.var(0),)))),
                Expr("*", (Expr.const(1), Expr("cube", (Expr.var(1),)))),
                Expr("*", (Expr.const(2), Expr("square", (Expr.var(2),)))),
            ]
            return [self._sum_expr(terms)]
        if self.problem.name == "Wave2D":
            exponent = Expr(
                "+",
                (
                    Expr("square", (Expr.var(1),)),
                    Expr("*", (Expr.const(0), Expr.var(0))),
                ),
            )
            return [Expr("*", (Expr("exp", (exponent,)), Expr("sin", (Expr.var(2),)))), Expr("*", (Expr("exp", (exponent,)), Expr("cos", (Expr.var(2),))))]
        if self.problem.name == "Wave3D":
            exponent = Expr(
                "+",
                (
                    Expr("square", (Expr.var(1),)),
                    Expr("-", (Expr("square", (Expr.var(3),)), Expr("*", (Expr.const(0), Expr.var(0))))),
                ),
            )
            return [Expr("*", (Expr("exp", (exponent,)), Expr("cos", (Expr.var(2),))))]
        return []

    def _record_candidate(self, candidate: Candidate, simulation: int) -> None:
        candidate = self._simplify_constants(candidate)
        candidate = self._archive_crossover(candidate)




        if self.best is None or candidate.reward > self.best.reward:
            self.best = candidate
        self.top_candidates.append(candidate)
        self.top_candidates.sort(key=lambda c: c.reward, reverse=True)
        self.top_candidates = self.top_candidates[: self.config.keep_top]
        if simulation % self.config.responsibility_interval == 0 or candidate is self.best:
            rho, edges = self._responsibility(candidate)
        else:
            rho, edges = 0.0, []
        self.archive.consider(candidate, rho, edges)

    def _backprop(self, path: Sequence[MCTSNode], reward: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += reward
            node.recent_rewards.append(reward)

    def fit(self) -> Candidate:
        if self.config.bootstrap_templates:
            for expression in self._structural_bootstrap():
                state = SearchState((), ())
                candidate = self._evaluate_terminal(SearchState(tuple(self._expr_to_actions(expression)), ()))
                self.bootstrap_candidates += 1
                self._record_candidate(candidate, 0)
        for simulation in range(1, self.config.simulations + 1):
            node = self.root
            path = [node]
            while node.state.frontier and len(node.children) == len(self.allowed_actions(node.state.frontier[0])):
                action = self._select_existing(node)
                node = node.children[action]
                path.append(node)

            if node.state.frontier:
                action = self._select_unexpanded(node)
                child = MCTSNode(self._next_state(node.state, action), node, action)
                node.children[action] = child
                node = child
                path.append(node)

            terminal = self._rollout(node.state)
            candidate = self._evaluate_terminal(terminal)
            self._record_candidate(candidate, simulation)
            self._backprop(path, candidate.reward)

        if self.best is None:
            raise RuntimeError("SED-MCTS finished without a terminal candidate")
        return self.best

    @staticmethod
    def _expr_to_actions(expr: Expr) -> List[str]:
        actions: List[str] = []

        def visit(node: Expr) -> None:
            if node.op.startswith("v") or node.op in {"C"}:
                actions.append(node.op)
                return
            if node.op == "L":


                actions.append("C")
                return
            actions.append(node.op)
            for child in node.children:
                visit(child)

        visit(expr)
        return actions

    def evaluate_on_split(self, candidate: Candidate) -> Dict[str, float]:
        pred = candidate.expr.evaluate(self.problem.x_eval, candidate.params).value
        residual = self.problem.residual(candidate.expr, candidate.params, self.problem.x_eval, self.problem.source_eval)
        return {
            "data_mse_eval": float(np.mean((pred - self.problem.y_eval) ** 2)),
            "phys_mse_eval": float(np.mean(residual**2)),
            "data_rmse_eval": float(np.sqrt(np.mean((pred - self.problem.y_eval) ** 2))),
        }

    def report(self, candidate: Candidate) -> Dict[str, object]:
        split = self.evaluate_on_split(candidate)
        return {
            "equation": self.problem.name,
            "target_description": self.problem.target_description,
            "expression": candidate.expr.render(candidate.params),
            "canonical_subtree": candidate.expr.canonical(),
            "constants": candidate.params.tolist(),
            "complexity": candidate.complexity,
            "data_mse_observation": candidate.data_mse,
            "phys_mse_residual": candidate.phys_mse,
            "normalized_loss": candidate.loss,
            "reward": candidate.reward,
            **split,
            "simulations": self.config.simulations,
            "terminal_evaluations": self.terminal_evaluations,
            "intervention_count": self.intervention_count,
            "archive_crossovers": self.archive_crossovers,
            "bootstrap_candidates": self.bootstrap_candidates,
            "archive_size": self.archive.size,
            "responsibility_edges": len(self.memory.edges),
            "bootstrap_operator_prior": self.bootstrap_prior,
            "memory": self.memory.summary(),
            "archive": self.archive.jsonable(),
        }


def run_one(
    name: str,
    config: SearchConfig,
    data_seed: int = 0,
    n_obs: int = 72,
    n_phys: int = 128,
    n_eval: int = 2048,
) -> Dict[str, object]:
    problem = make_problem(name, seed=data_seed, n_obs=n_obs, n_phys=n_phys, n_eval=n_eval)
    search = SEDMCTS(problem, config)
    best = search.fit()
    return search.report(best)


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
