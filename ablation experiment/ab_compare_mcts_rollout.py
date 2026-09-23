






import argparse
import os
import sys
import time
import numpy as np
import h5py

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


def format_preview(arr, n=8):
    arr = np.asarray(arr).reshape(-1)
    n = min(n, arr.shape[0])
    return np.array2string(arr[:n], precision=4, separator=", ")


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

        elapsed = time.time() - t0
        return {
            "mode": mode,
            "seed": seed,
            "status": "ok",
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "evals": int(evaluations),
            "time_s": elapsed,
            "expr": str(sym_exp),
            "path": str(path),
            "error": "",
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "mode": mode,
            "seed": seed,
            "status": "fail",
            "mse": float("inf"),
            "mae": float("inf"),
            "r2": float("nan"),
            "evals": -1,
            "time_s": elapsed,
            "expr": "<none>",
            "path": "<none>",
            "error": str(e),
        }


def print_table(rows):
    print("\n" + "=" * 130)
    print("A/B Comparison (random rollout vs guided rollout) - Advection")
    print("=" * 130)
    print(f"{'mode':<10} {'seed':<6} {'status':<8} {'mse':<14} {'mae':<12} {'r2':<10} {'evals':<8} {'time(s)':<9} expr")
    print("-" * 130)

    for r in rows:
        r2_str = f"{r['r2']:.4f}" if np.isfinite(r["r2"]) else "nan"
        print(
            f"{r['mode']:<10} {r['seed']:<6} {r['status']:<8} {r['mse']:<14.6e} "
            f"{r['mae']:<12.6e} {r2_str:<10} {r['evals']:<8} {r['time_s']:<9.2f} {r['expr']}"
        )
        if r["status"] != "ok":
            print(f"{'':<10} {'':<6} {'':<8} error: {r['error']}")

    random_rows = [r for r in rows if r["mode"] == "random" and r["status"] == "ok"]
    guided_rows = [r for r in rows if r["mode"] == "guided" and r["status"] == "ok"]

    print("-" * 130)
    if random_rows:
        print(
            f"AVG random: mse={np.mean([r['mse'] for r in random_rows]):.6e}, "
            f"mae={np.mean([r['mae'] for r in random_rows]):.6e}, "
            f"r2={np.mean([r['r2'] for r in random_rows]):.4f}, "
            f"time={np.mean([r['time_s'] for r in random_rows]):.2f}s"
        )
    else:
        print("AVG random: no successful runs")

    if guided_rows:
        print(
            f"AVG guided: mse={np.mean([r['mse'] for r in guided_rows]):.6e}, "
            f"mae={np.mean([r['mae'] for r in guided_rows]):.6e}, "
            f"r2={np.mean([r['r2'] for r in guided_rows]):.4f}, "
            f"time={np.mean([r['time_s'] for r in guided_rows]):.2f}s"
        )
    else:
        print("AVG guided: no successful runs")

    if random_rows and guided_rows:
        random_mse = np.mean([r["mse"] for r in random_rows])
        guided_mse = np.mean([r["mse"] for r in guided_rows])
        if np.isfinite(random_mse) and np.isfinite(guided_mse) and random_mse > 1e-12:
            gain = (random_mse - guided_mse) / random_mse * 100.0
            print(f"MSE improvement (guided vs random): {gain:.2f}%")

    print("=" * 130 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run A/B comparison for MCTS rollout modes (Advection).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds for repeated runs.")
    parser.add_argument("--max_expressions", type=int, default=1200, help="Expression budget per run.")
    parser.add_argument("--verbose", action="store_true", help="Print internal MCTS logs.")
    parser.add_argument("--bootstrap_random_expressions", type=int, default=0, help="Offline random expressions for prior bootstrap.")
    parser.add_argument("--bootstrap_topk", type=int, default=0, help="Top-k expressions used to initialize prior (0 means top 20%%).")
    args = parser.parse_args()

    X_flat, y_flat = load_xy()
    print(f"Loaded Advection data: X={X_flat.shape}, y={y_flat.shape}")
    print(f"Seeds={args.seeds}, max_expressions={args.max_expressions}")
    print(f"Bootstrap prior: random_expressions={args.bootstrap_random_expressions}, topk={args.bootstrap_topk}")

    print("\n[Ground Truth Summary]")
    print(f"  y_true mean={np.mean(y_flat):.6e}, std={np.std(y_flat):.6e}, min={np.min(y_flat):.6e}, max={np.max(y_flat):.6e}")
    print(f"  y_true preview: {format_preview(y_flat, n=12)}")
    print(f"  x preview: {format_preview(X_flat[0], n=8)}")
    print(f"  t preview: {format_preview(X_flat[1], n=8)}")

    rows = []
    for seed in args.seeds:
        print(f"\n[Seed {seed}] A: random rollout")
        rows.append(run_once(
            X_flat, y_flat, seed, guided=False, max_expr=args.max_expressions, verbose=args.verbose,
            bootstrap_n=args.bootstrap_random_expressions, bootstrap_topk=args.bootstrap_topk
        ))

        print(f"[Seed {seed}] B: guided rollout")
        rows.append(run_once(
            X_flat, y_flat, seed, guided=True, max_expr=args.max_expressions, verbose=args.verbose,
            bootstrap_n=args.bootstrap_random_expressions, bootstrap_topk=args.bootstrap_topk
        ))

    print_table(rows)


if __name__ == "__main__":
    main()
