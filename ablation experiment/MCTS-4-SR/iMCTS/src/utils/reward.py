import numpy as np
from typing import Tuple, Callable, Any
try:
    import numba as _numba
except Exception:
    _numba = None

try:
    import nlopt as _nlopt
except Exception:
    _nlopt = None

def _cal_res_numpy(y_pred: np.ndarray, y_train: np.ndarray, sigma: float) -> float:




    diff = np.asarray(y_pred) - np.asarray(y_train)
    if diff.size == 0:
        return 0.0

    mse = np.dot(diff, diff) / diff.size

    if sigma == 0.0:
        sigma = 1.0
    res1 = np.sqrt(mse) / sigma


    if np.isnan(res1) or np.isinf(res1):
        return 0.0
    return 1.0 / (1.0 + res1)


if _numba is not None:
    _cal_res_jitted = _numba.njit(fastmath=True, cache=True)(_cal_res_numpy)

    def cal_res_numba(y_pred: np.ndarray, y_train: np.ndarray, sigma: float) -> float:
        try:
            return float(_cal_res_jitted(y_pred, y_train, sigma))
        except Exception:
            return _cal_res_numpy(y_pred, y_train, sigma)
else:
    cal_res_numba = _cal_res_numpy

class Optimizer:









    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, context: dict[str, Any],
                 cal_reward: Callable | None = None, optimization_method: str = 'LN_NELDERMEAD'):
        self.x_train = x_train
        self.y_train = y_train

        self.sigma = float(np.std(y_train)) if y_train is not None else 1.0
        self.context = context

        self.cal_reward = cal_reward if cal_reward is not None else self._cal_res

        self.optimization_method = (
            getattr(_nlopt, optimization_method, None)
            if _nlopt is not None else None
        )

    def optimize_constants(self, state) -> Tuple[str, float]:







        expression: str = state.get_expression()


        if any(tok in expression for tok in ("zoo", "nan", "inf")):
            if state.constant_count > 0:
                return expression, 0.0
            return expression, 0.0

        constant_count = state.constant_count
        if constant_count > 0:

            r_len = state.real_constant_count
            c_len = constant_count - r_len



            f_pred_const = eval(compile('lambda x, C, R: ' + expression, '<expr>', 'eval'), self.context)



            if c_len:
                guess_c = np.random.randn(c_len * 2)
            else:
                guess_c = np.empty(0)
            if r_len:
                guess_r = np.random.randn(r_len)
            else:
                guess_r = np.empty(0)
            initial_guess = np.concatenate((guess_c, guess_r)) if constant_count else np.empty(0)
            n_params = len(initial_guess)

            if _nlopt is not None and self.optimization_method is not None:

                opt = _nlopt.opt(self.optimization_method, n_params)
                opt.set_min_objective(
                    lambda p, grad: -self._cal_reward_wrapper(p, f_pred_const, c_len)
                )
                opt.set_xtol_rel(1e-6)
                opt.set_maxeval(250)
                bounds = np.array([-10.0] * n_params)
                opt.set_lower_bounds(bounds)
                opt.set_upper_bounds(-bounds)

                try:
                    optimized_params = opt.optimize(initial_guess)
                except Exception as e:
                    print(f"An unexpected error occurred during NLopt optimization: {e}")
                    return expression, 0.0
            else:

                try:
                    from scipy.optimize import minimize

                    result = minimize(
                        lambda p: -self._cal_reward_wrapper(p, f_pred_const, c_len),
                        initial_guess,
                        method="Nelder-Mead",
                        bounds=[(-10.0, 10.0)] * n_params,
                        options={"maxiter": 250},
                    )
                    optimized_params = np.asarray(result.x, dtype=float)
                except Exception as e:
                    print(f"Constant optimization fallback failed: {e}")
                    optimized_params = initial_guess


            if c_len:
                complex_constants = optimized_params[:c_len] + 1j * optimized_params[c_len:2 * c_len]
            else:
                complex_constants = []
            real_constants = optimized_params[2 * c_len:]



            for idx, c in enumerate(complex_constants):
                expression = expression.replace(f'C[{idx}]', repr(c))
            for idx, r in enumerate(real_constants):
                expression = expression.replace(f'R[{idx}]', repr(float(r)))

        elif constant_count == 0:

            pass


        try:
            f_pred = eval(compile(f"lambda x: {expression}", "<expr-final>", "eval"), self.context)
        except Exception:
            return expression, 0.0


        try:

            with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
                reward = self.cal_reward(self.x_train, self.y_train, f_pred)

            reward = float(reward)
            if not np.isfinite(reward):
                reward = 0.0
        except ZeroDivisionError:
            reward = 0.0
        except Exception:

            reward = 0.0
        return expression, float(reward)

    def _cal_res(self, x_train: np.ndarray, y_train: np.ndarray, f_pred: Callable[[np.ndarray], np.ndarray]) -> float:




        try:
            y_pred = f_pred(x_train)

            if not isinstance(y_pred, np.ndarray):
                y_pred = np.asarray(y_pred)
            return float(cal_res_numba(y_pred, y_train, self.sigma))
        except ZeroDivisionError:
            return 0.0
        except Exception:

            return 0.0

    def _cal_reward_wrapper(self, params: np.ndarray, f_pred_const: Callable, c_len: int) -> float:





        if c_len:
            complex_params = params[:c_len] + 1j * params[c_len:2 * c_len]
            real_params = params[2 * c_len:]
        else:
            complex_params = []
            real_params = params

        reward = self.cal_reward(self.x_train, self.y_train,
                                 lambda x: f_pred_const(x, C=complex_params, R=real_params))
        return float(reward)
