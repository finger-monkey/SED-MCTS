









from __future__ import annotations

import numpy as np

from prs_iMCTS import Regressor


def main() -> None:




    np.random.seed(0)
    var_num = 1
    n_samples = 20
    X = np.random.uniform(0.0, 2.0, (var_num, n_samples))


    def f(x: np.ndarray) -> np.ndarray:
        return np.log(x[0] + 1.0) + np.log(x[0] ** 2 + 1.0)

    Y = f(X)


    ops = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]




    model = Regressor(
        x_train=X,
        y_train=Y,
        ops=ops,
        verbose=True,
        optimization_method="LN_LBFGS",
        max_expressions=200000,
        rollout_guided=True,
        prior_bootstrap_random_expressions=300,
        prior_bootstrap_topk=30,
        enable_two_stage_refine=False,


        refine_guided_bootstrap=True,
        refine_guided_bootstrap_warmup_expressions=200,
        refine_guided_bootstrap_affinity=0.55,
        refine_guided_bootstrap_max_buffer=256,


        refine_stat_feedback_weight=0.0,
        refine_ablation_feedback_weight=0.0,





    )

    sym_exp, vec_exp, evaluations, path = model.fit(seed=0)
    print("\n===== Demo finished =====")
    print(f"evaluations={evaluations}")
    print(f"best simplified expression: {sym_exp}")
    print(f"best raw vector expression: {vec_exp}")
    print(f"best path (symbols): {path}")


if __name__ == "__main__":
    main()

