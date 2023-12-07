# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.stats import wasserstein_distance
from wlpy.regression import LocLin
from wlpy.gist import current_time
from utils import *


def cal_y_signal(eval, params):
    return (
        np.sqrt(2)
        + np.dot(eval, params)
        + np.cos(eval[:, -1] * 3)
        + 0.5 * eval[:, -1] ** 2
    )


N = 1000  # sample size
M = 200  # number of draws from intervals
K = 4  # number of covariates
beta = np.ones(K) * np.pi / 10
var_epsilon = 1.0
interval_bias = [0.0, 0.0]
x_distri = "uniform"
epsilon_distri = "normal"
df = 2
n_test_points = 100
scale = 2.0
p1, p2 = 1, 1

tolerance = 1 / N

x = gen_x(N, K, x_distri)
epsilon = gen_noise(N, var_epsilon, epsilon_distri, df)
y, yl, yu, y_middle = gen_outcomes(
    get_interval, cal_y_signal, beta, interval_bias, x, epsilon, scale
)

evaluation_points, y_eval_signal = gen_eval(cal_y_signal, K, beta, n_test_points, x)

params_dict = {
    "N": N,
    "M": M,
    "K": K,
    "beta": beta,
    "var_epsilon": var_epsilon,
    "interval_bias": interval_bias,
    "x_distri": x_distri,
    "epsilon_distri": epsilon_distri,
    "df": df,
    "tolerance": tolerance,
    "n_test_points": n_test_points,
    "evaluation_points": evaluation_points,
    "y_eval_signal": y_eval_signal,
    "scale": scale,
}
# %%

weights, yd = calculate_weights_and_yd(M, N, yl, yu, 1, 1)

(
    x_train,
    x_test,
    yu_train,
    yu_test,
    yl_train,
    yl_test,
    y_train,
    y_test,
    yd_train,
    yd_test,
) = train_test_split(x, yu, yl, y, yd.T, test_size=0.5, random_state=42)


# %%
def fit_and_predict(
    method, x_train, y_train, x_test, evaluation_points, krr_kernel="rbf"
):
    if method not in ["loclin", "linear", "rf", "krr"]:
        raise ValueError("method must be one of 'loclin', 'linear' or 'rf'")
    if method == "loclin":
        model = [LocLin(x_train, j) for j in yd_train.T]
        y_test_pred = get_loclin_pred(x_test, model)
        y_eval_pred = get_loclin_pred(evaluation_points, model)
        y_mid_fit = LocLin(x, y_middle).vec_fit(evaluation_points)
        y_true_fit = LocLin(x, y).vec_fit(evaluation_points)
        yl_fit, yu_fit = None, None

    else:
        if method == "linear":
            model = LinearRegression()
        if method == "rf":
            model = RandomForestRegressor(n_estimators=200, max_depth=10)
        if method == "krr":
            model = KernelRidge(alpha=0.1, kernel=krr_kernel)

        model.fit(x_train, y_train)
        y_test_pred = model.predict(x_test).T
        y_eval_pred = model.predict(evaluation_points).T
        y_mid_fit = model.fit(x, y_middle).predict(evaluation_points)
        y_true_fit = model.fit(x, y).predict(evaluation_points)
        yl_fit = model.fit(x, yl).predict(evaluation_points)
        yu_fit = model.fit(x, yu).predict(evaluation_points)
    return y_test_pred, y_eval_pred, y_mid_fit, y_true_fit, yl_fit, yu_fit


method = "krr"

y_test_pred, y_eval_pred, y_mid_fit, y_true_fit, yl_fit, yu_fit = fit_and_predict(
    method, x_train, yd_train, x_test, evaluation_points
)
indices = select_indices(compute_score, tolerance, yu_test, yl_test, y_test_pred)
# Plotting Results
plot_result(
    get_interval,
    **params_dict,
    indices=indices,
    y_eval_pred=y_eval_pred,
    y_mid_fit=y_mid_fit,
    y_true_fit=y_true_fit,
    # yl_fit=yl_fit,
    # yu_fit=yu_fit,
    filename=f"simulation-results/{current_time()}-{method}-{N}-{M}-{K}.pdf",
)


# %%
# This part compares the effect of M on the closeness of drawed smaple, both unconditonally and conditionally.
# Unconditonally
x = gen_x(N, K, x_distri)
x_fixed = np.zeros([N, K])

epsilon = gen_noise(N, var_epsilon, epsilon_distri, df)
y, yl, yu, y_middle = gen_outcomes(
    get_interval, cal_y_signal, beta, interval_bias, x, epsilon, scale=2.0
)
M = 2000
weights, yd = calculate_weights_and_yd(M, N, yl, yu, 1, 1)
plt.hist([y, yd[0]], density=True, bins=50)
p1, p2 = 1, 1
scale = 1.0
# for scale in (1.0, 2.0, 3.0, 4.0):
# for M in (10, 200, 2000, 10000):
for p1 in (0.5, 1, 2):
    for p2 in (0.5, 1, 2):
        x = gen_x(N, K, x_distri)
        x_fixed = np.zeros([N, K])

        epsilon = gen_noise(N, var_epsilon, epsilon_distri, df)
        y, yl, yu, y_middle = gen_outcomes(
            get_interval,
            cal_y_signal,
            beta,
            interval_bias,
            x_fixed,
            epsilon,
            scale=scale,
        )

        weights, yd = calculate_weights_and_yd(M, N, yl, yu, p1, p2)
        plt.hist([yd[0], y], density=True, bins=50)
        result = np.array([wasserstein_distance(yd[i], y) for i in range(M)])
        # print(p1, p2, result.mean(), result.min())
        print(f"{p1} {p2} {result.mean():.3f} {result.min():.3f}")
# %%
import scipy.stats as stats

yd_sorted = np.sort(yd[0])
y_sorted = np.sort(y)
plt.plot(y_sorted, yd_sorted)
plt.plot(y_sorted, y_sorted)
print(yd[0])
# %%
# conditional on fixed x
x_fixed = np.zeros([N, K])
epsilon = gen_noise(N, var_epsilon, epsilon_distri, df)
y, yl, yu, y_middle = gen_outcomes(
    get_interval, cal_y_signal, beta, interval_bias, x_fixed, epsilon, scale=2.0
)
p1, p2 = 2, 0.5
weights, yd = calculate_weights_and_yd(M, N, yl, yu, p1, p2)
plt.figure()
plt.hist([y, yd[1], yl], density=True, bins=50, label=["y", "yd", "yl"])
plt.title(
    f"Conditional on fixed x, when p1={p1}, p2={p2}",
)
plt.legend()
# plt.savefig(f"simulation-results/{current_time()}-conditional-{p1}-{p2}.pdf")
plt.show()
dw = np.array([wasserstein_distance(yd[i], y) for i in range(M)])
print(dw.mean())
# %%
