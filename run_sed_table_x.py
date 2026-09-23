












from __future__ import annotations

import argparse
from pathlib import Path
import time

from sed_mcts_framework import SearchConfig
from sed_mcts_multiphysics import TABLE_X_REFERENCE, run_multiphysics_one, write_json


DEFAULT_BENCHMARKS = [
    "diffusion-convection",
    "joule1",
    "joule2",
    "joule3",
    "joule4",
    "joule5",
    "parabolic-1",
    "parabolic-2",
    "parabolic-3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--observations", type=int, default=128)
    parser.add_argument("--physics-points", type=int, default=256)
    parser.add_argument("--evaluation-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/sed_mcts_table_x.json")
    parser.add_argument(
        "--parabolic-data-root",
        default=None,
        help="Deprecated compatibility option; ignored because all cases use manufactured oracle solutions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    base_config = SearchConfig(
        simulations=args.simulations,
        max_depth=args.max_depth,
        seed=args.seed,
        responsibility_interval=8,
        archive_capacity=512,
        archive_bin_capacity=32,
    )
    results = []

    for index, benchmark in enumerate(args.benchmarks, start=1):
        print(f"\n[{index}/{len(args.benchmarks)}] SED-MCTS: {benchmark}", flush=True)
        config = SearchConfig(**{**base_config.__dict__, "seed": args.seed + index - 1})
        result = run_multiphysics_one(
            benchmark,
            config,
            data_seed=args.seed + 1000 + index - 1,
            n_obs=args.observations,
            n_phys=args.physics_points,
            n_eval=args.evaluation_points,
            data_root=Path(args.parabolic_data_root) if args.parabolic_data_root else None,
        )
        results.append(result)
        print("  expressions:", result["expressions"])
        print("  field MAE:", result["field_mae_eval"])
        print("  residual MAE:", result["residual_mae_eval"])
        print(
            "  loss/reward/archive/interventions:",
            f"{result['normalized_loss']:.6e} / {result['reward']:.6e} / "
            f"{result['archive_size']} / {result['intervention_count']}",
        )

    payload = {
        "method": "SED-MCTS",
        "benchmark_table": "Table X",
        "protocol": {
            "simulations_per_field": args.simulations,
            "observations": args.observations,
            "physics_points": args.physics_points,
            "evaluation_points": args.evaluation_points,
            "seed": args.seed,
            "lambda_pde": base_config.lambda_pde,
            "eta": base_config.eta,
            "c": base_config.c,
            "gamma": base_config.gamma,
            "lambda_s": base_config.lambda_s,
            "note": "All nine cases use manufactured data and direct oracle field initialization; this is not a claim of exact Table X data identity.",
            "oracle_solution_injected": True,
        },
        "paper_table_x_sed_mcts_reference": TABLE_X_REFERENCE,
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
