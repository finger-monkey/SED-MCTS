











from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from sed_mcts_framework import SearchConfig, run_one, write_json


PAPER_TABLE_V = {
    "Advection": {"mean": 8.1e-14, "std": 3.2e-14},
    "Diffusion": {"mean": 4.3e-4, "std": 1.8e-4},
    "Poisson2D": {"mean": 2.8e-6, "std": 1.3e-6},
    "Poisson3D": {"mean": 8.1e-6, "std": 3.5e-6},
    "Wave2D": {"mean": 3.4e-3, "std": 1.5e-3},
    "Wave3D": {"mean": 4.9e-3, "std": 2.1e-3},
}


def operator_set_for(equation: str) -> tuple[str, ...] | None:


    key = equation.lower()
    if key in {"advection", "adv"}:
        return ("neg", "sin", "cos", "+", "-", "*")
    if key in {"diffusion", "diffusion-reaction", "diffusion_reaction"}:
        return ("neg", "+", "-", "*", "exp", "sin", "cos")
    if key in {"poisson2d", "poisson"}:
        return ("neg", "+", "-", "*", "square", "cube", "fourth")
    if key in {"poisson3d", "poisson_3d"}:
        return ("neg", "+", "-", "*", "square", "cube", "fourth")
    if key in {"wave2d", "wave"}:
        return ("neg", "+", "-", "*", "exp", "sin", "cos", "square")
    if key in {"wave3d", "wave_3d"}:
        return ("neg", "+", "-", "*", "exp", "sin", "cos", "square")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equations",
        nargs="+",
        default=["advection", "diffusion", "poisson2d", "poisson3d", "wave2d", "wave3d"],
        help="Table V equations to run.",
    )
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--observations", type=int, default=72)
    parser.add_argument("--physics-points", type=int, default=128)
    parser.add_argument("--evaluation-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/sed_mcts_table_v_representative.json",
        help="JSON result path, relative to this script directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SearchConfig(
        simulations=args.simulations,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    results = []
    started = time.perf_counter()

    for offset, equation in enumerate(args.equations):
        print(f"\n[{offset + 1}/{len(args.equations)}] SED-MCTS: {equation}", flush=True)

        config_for_run = SearchConfig(
            **{
                **config.__dict__,
                "seed": args.seed + offset,
                "operator_set": operator_set_for(equation),
            }
        )
        problem_result = run_one(
            equation,
            config_for_run,
            data_seed=args.seed + 1000 + offset,
            n_obs=args.observations,
            n_phys=args.physics_points,
            n_eval=args.evaluation_points,
        )
        paper_name = problem_result["equation"]
        target = PAPER_TABLE_V.get(paper_name)
        if target:
            problem_result["paper_table_v_reference"] = target
            problem_result["eval_mse_over_paper_mean"] = problem_result["data_mse_eval"] / target["mean"]
        results.append(problem_result)
        print(f"  expression: {problem_result['expression']}")
        print(f"  observation MSE: {problem_result['data_mse_observation']:.6e}")
        print(f"  dense eval MSE:  {problem_result['data_mse_eval']:.6e}")
        print(f"  PDE residual MSE: {problem_result['phys_mse_eval']:.6e}")
        print(f"  loss/reward:      {problem_result['normalized_loss']:.6e} / {problem_result['reward']:.6e}")
        print(
            "  complexity/archive/interventions: "
            f"{problem_result['complexity']} / {problem_result['archive_size']} / "
            f"{problem_result['intervention_count']}"
        )

    payload = {
        "method": "SED-MCTS",
        "protocol": {
            "lambda_pde": config.lambda_pde,
            "eta": config.eta,
            "epsilon": config.epsilon,
            "c": config.c,
            "gamma": config.gamma,
            "lambda_s": config.lambda_s,
            "K": 5,
            "observations": args.observations,
            "physics_points": args.physics_points,
            "evaluation_points": args.evaluation_points,
            "simulations": args.simulations,
            "seed": args.seed,
            "note": "Independent closed-form splits; not a claim of exact Table V protocol identity.",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    write_json(output, payload)
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
