



import argparse
import csv
import os
import sys
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
mcts_dir = os.path.join(script_dir, "MCTS-4-SR")
sys.path.insert(0, mcts_dir)
sys.path.insert(0, script_dir)

from iMCTS import Regressor


class Config:
    DATA_FILENAME = "1D_Advection_Sols_beta1.0_Num200.hdf5"
    SAMPLE_IDX = 0
    TIME_SUBSAMPLE = 20
    SPACE_SUBSAMPLE = 20
    OPS = ["+", "-", "*", "/", "sin", "cos"]


def find_file(name, start_dir):
    for root, _, files in os.walk(start_dir):
        if name in files:
            return os.path.join(root, name)
    return None


def load_xy():


    data_path = find_file(Config.DATA_FILENAME, script_dir)
    if not data_path:
        raise FileNotFoundError(f"Data file not found: {Config.DATA_FILENAME}")

    with h5py.File(data_path, "r") as f:
        u_tensor = f["tensor"][Config.SAMPLE_IDX]
        x_coord = f["x-coordinate"][:]
        t_coord = f["t-coordinate"][:]

    if u_tensor.shape[0] < len(t_coord):
        t_coord = t_coord[: u_tensor.shape[0]]

    x_sub = x_coord[:: Config.SPACE_SUBSAMPLE]
    t_sub = t_coord[:: Config.TIME_SUBSAMPLE]
    u_sub = u_tensor[:: Config.TIME_SUBSAMPLE, :: Config.SPACE_SUBSAMPLE]

    T, X = np.meshgrid(t_sub, x_sub, indexing="ij")
    X_flat = np.vstack([X.ravel(), T.ravel()])
    y_flat = u_sub.ravel()
    return X_flat, y_flat


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else 1.0 - (ss_res / ss_tot)
    return mse, mae, r2


def run_once(X_flat, y_flat, seed, guided, max_expr, verbose, bootstrap_n, bootstrap_topk):
    mode = "guided" if guided else "random"
    model = Regressor(
        x_train=X_flat,
        y_train=y_flat,
        ops=Config.OPS,
        max_expressions=max_expr,
        verbose=verbose,
        rollout_guided=guided,
        prior_bootstrap_random_expressions=bootstrap_n,
        prior_bootstrap_topk=bootstrap_topk,
    )

    t0 = time.time()
    try:
        sym_exp, vec_exp, evaluations, path = model.fit(seed=seed)
        y_pred = model.predict(X_flat, vec_exp)
        mse, mae, r2 = compute_metrics(y_flat, y_pred)
        return {
            "status": "ok", "mode": mode, "seed": seed,
            "bootstrap_n": bootstrap_n, "bootstrap_topk": bootstrap_topk,
            "max_expressions": max_expr, "mse": mse, "mae": mae, "r2": r2,
            "evals": int(evaluations), "time_s": time.time() - t0,
            "expr": str(sym_exp), "path": str(path), "error": "",
        }
    except Exception as e:
        return {
            "status": "fail", "mode": mode, "seed": seed,
            "bootstrap_n": bootstrap_n, "bootstrap_topk": bootstrap_topk,
            "max_expressions": max_expr, "mse": float("inf"), "mae": float("inf"), "r2": float("nan"),
            "evals": -1, "time_s": time.time() - t0, "expr": "<none>", "path": "<none>", "error": str(e),
        }


def aggregate(detailed_rows):
    grouped = {}
    for row in detailed_rows:
        key = (row["bootstrap_n"], row["bootstrap_topk"])
        grouped.setdefault(key, {"guided": [], "random": []})
        if row["status"] == "ok":
            grouped[key][row["mode"]].append(row)

    summary_rows = []
    for (n, k), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        guided = vals["guided"]
        random_rows = vals["random"]

        def avg(arr, name):
            if not arr:
                return float("nan")
            return float(np.mean([r[name] for r in arr]))

        guided_mse = avg(guided, "mse")
        random_mse = avg(random_rows, "mse")
        improvement = float("nan")
        if np.isfinite(guided_mse) and np.isfinite(random_mse) and random_mse > 1e-12:
            improvement = (random_mse - guided_mse) / random_mse * 100.0

        summary_rows.append({
            "bootstrap_n": n,
            "bootstrap_topk": k,
            "guided_ok_runs": len(guided),
            "random_ok_runs": len(random_rows),
            "guided_mse_mean": guided_mse,
            "guided_mae_mean": avg(guided, "mae"),
            "guided_r2_mean": avg(guided, "r2"),
            "guided_time_mean": avg(guided, "time_s"),
            "random_mse_mean": random_mse,
            "random_mae_mean": avg(random_rows, "mae"),
            "random_r2_mean": avg(random_rows, "r2"),
            "random_time_mean": avg(random_rows, "time_s"),
            "mse_improvement_vs_random_percent": improvement,
        })

    return summary_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(summary_rows, z_key, title, output_path):
    ns = sorted({int(r["bootstrap_n"]) for r in summary_rows})
    ks = sorted({int(r["bootstrap_topk"]) for r in summary_rows})
    z = np.full((len(ns), len(ks)), np.nan)

    for r in summary_rows:
        i = ns.index(int(r["bootstrap_n"]))
        j = ks.index(int(r["bootstrap_topk"]))
        z[i, j] = float(r[z_key])

    fig, ax = plt.subplots(figsize=(1.8 * max(4, len(ks)), 1.2 * max(4, len(ns))))
    im = ax.imshow(z, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(np.arange(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    ax.set_xlabel("bootstrap_topk")
    ax.set_ylabel("bootstrap_random_expressions")
    ax.set_title(title)

    for i in range(len(ns)):
        for j in range(len(ks)):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Ablation of offline prior bootstrap (N, top-k).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to run.")
    parser.add_argument("--max_expressions", type=int, default=1200, help="MCTS expression budget per run.")
    parser.add_argument("--bootstrap_random_expressions_grid", type=int, nargs="+", default=[0, 1000, 3000, 4000])
    parser.add_argument("--bootstrap_topk_grid", type=int, nargs="+", default=[0, 50, 100, 300])
    parser.add_argument("--verbose", action="store_true", help="Verbose logs from solver.")
    parser.add_argument("--output_dir", type=str, default="", help="Optional output directory.")
    args = parser.parse_args()

    X_flat, y_flat = load_xy()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, "ablation_outputs", f"prior_bootstrap_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded Advection data: X={X_flat.shape}, y={y_flat.shape}")
    print(f"Seeds={args.seeds}, max_expressions={args.max_expressions}")
    print(f"bootstrap_random_expressions_grid={args.bootstrap_random_expressions_grid}")
    print(f"bootstrap_topk_grid={args.bootstrap_topk_grid}")
    print(f"Output dir: {output_dir}")

    detailed_rows = []
    for n in args.bootstrap_random_expressions_grid:
        for k in args.bootstrap_topk_grid:
            if n == 0 and k > 0:
                continue
            if n > 0 and k > n:
                continue

            print("\n" + "-" * 80)
            print(f"Config: bootstrap_n={n}, bootstrap_topk={k}")
            print("-" * 80)

            for seed in args.seeds:
                print(f"[seed={seed}] random rollout")
                detailed_rows.append(run_once(X_flat, y_flat, seed, False, args.max_expressions, args.verbose, n, k))
                print(f"[seed={seed}] guided rollout")
                detailed_rows.append(run_once(X_flat, y_flat, seed, True, args.max_expressions, args.verbose, n, k))

    summary_rows = aggregate(detailed_rows)

    detailed_csv = os.path.join(output_dir, "detailed_runs.csv")
    summary_csv = os.path.join(output_dir, "summary.csv")
    write_csv(detailed_csv, detailed_rows)
    write_csv(summary_csv, summary_rows)

    plot_heatmap(summary_rows, "guided_mse_mean", "Guided MSE vs bootstrap_n / topk", os.path.join(output_dir, "heatmap_guided_mse.png"))
    plot_heatmap(summary_rows, "mse_improvement_vs_random_percent", "MSE improvement (%) Guided over Random", os.path.join(output_dir, "heatmap_mse_improvement_vs_random.png"))

    print("\nDone.")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Summary CSV : {summary_csv}")
    print(f"Figure       : {os.path.join(output_dir, 'heatmap_guided_mse.png')}")
    print(f"Figure       : {os.path.join(output_dir, 'heatmap_mse_improvement_vs_random.png')}")


if __name__ == "__main__":
    main()
