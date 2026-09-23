from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from prs_iMCTS import Regressor, simplify_expression




PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
BASE_OPS = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "R"]


def ops_for_dataset(dataset: str) -> list[str]:

    name = dataset.lower()
    if name.startswith(("poisson", "heat")):
        return ["+", "-", "*", "R"]
    if name.startswith("wave"):
        return ["+", "-", "*", "/", "sin", "cos", "exp", "R"]
    return BASE_OPS


def max_depth_for_dataset(dataset: str) -> int:
    name = dataset.lower()
    if name.startswith(("poisson", "heat")):
        return 8
    if name.startswith("wave"):
        return 7
    return 6


def max_constants_for_dataset(dataset: str) -> int:
    name = dataset.lower()
    if name.startswith(("poisson", "heat")):
        return 10
    if name.startswith("wave"):
        return 8
    return 5


@dataclass
class Result:
    dataset: str
    method: str
    seed: int
    budget: int
    expression: str
    simplified_expression: str
    train_mse: float
    train_rmse: float
    train_mae: float
    train_r2: float
    train_nrmse: float
    method_score: float
    evaluations: int
    elapsed_sec: float
    status: str
    error: str = ""


def load_dataset(path: Path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    rng = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / rng) if rng > 0 else float("nan")
    return mse, rmse, mae, r2, nrmse


def expression_to_prediction(expression: str, x: np.ndarray) -> np.ndarray:
    import sympy as sp

    expr_text = expression.replace("^", "**")
    local_dict = {f"x{i}": sp.Symbol(f"x{i}") for i in range(x.shape[1])}
    local_dict.update({"sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log})
    expr = sp.sympify(expr_text, locals=local_dict)
    fn = sp.lambdify([local_dict[f"x{i}"] for i in range(x.shape[1])], expr, modules="numpy")
    values = fn(*[x[:, i] for i in range(x.shape[1])])
    if np.isscalar(values):
        values = np.full(x.shape[0], float(values), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        x, y = load_dataset(DATA_DIR / args.dataset)
        ops = ops_for_dataset(args.dataset)
        model = Regressor(
            x_train=x.T.astype(np.float64),
            y_train=y.astype(np.float64),
            ops=ops,
            max_depth=max_depth_for_dataset(args.dataset),
            K=500,
            c=4.0,
            gamma=0.5,
            gp_rate=0.2,
            mutation_rate=0.1,
            exploration_rate=0.2,
            max_constants=max_constants_for_dataset(args.dataset),
            max_expressions=args.budget,
            verbose=False,
            optimization_method="LD_LBFGS",
            rollout_guided=True,
            rollout_epsilon=0.15,
            rollout_temperature=1.0,
            prior_bootstrap_random_expressions=1500,
            prior_bootstrap_topk=30,
            enable_two_stage_refine=True,
            refine_topk_subtrees=3,
            refine_topm_actions=3,
            refine_every_n=2,
            refine_guided_bootstrap=True,
            refine_guided_bootstrap_warmup_expressions=1000,
            refine_guided_bootstrap_affinity=0.55,
            reverse_search_enabled=False,
        )
        sym_exp, raw_exp, evaluations, _ = model.fit(seed=args.seed)
        expression = str(raw_exp)
        simplified = simplify_expression(str(sym_exp))
        y_pred = expression_to_prediction(simplified, x)
        selected_metrics = metrics(y, np.where(np.isfinite(y_pred), y_pred, np.nanmean(y)))
        mse, rmse, mae, r2, nrmse = selected_metrics
        result = Result(args.dataset, "prs-mcts", args.seed, args.budget, expression, simplified, mse, rmse, mae, r2, nrmse, float("nan"), int(evaluations), time.perf_counter() - start, "ok")
    except Exception as exc:
        result = Result(args.dataset, "prs-mcts", args.seed, args.budget, "", "", float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, time.perf_counter() - start, "failed", repr(exc))
    print(json.dumps(asdict(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
