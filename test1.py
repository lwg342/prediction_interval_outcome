# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.stats import wasserstein_distance
from wlpy.regression import LocLin
from wlpy.gist import current_time
from utils import *


# %%


def cal_y_signal(eval, params):
    return np.sqrt(2) + np.dot(eval, params)  # + 0.2 * eval[:, -1] ** 2
    # + np.cos(eval[:, -1])


N = 2000  # sample size
M = 100  # number of draws from intervals
K = 3  # number of covariates
beta = np.ones(K) * np.pi / 10
var_epsilon = 1.0
interval_bias = [0.0, 0.0]
x_distri = "uniform"
epsilon_distri = "normal"
df = 2
n_test_points = 100

tolerance = 1/N

x = gen_x(N, K, x_distri)
epsilon = gen_noise(N, var_epsilon, epsilon_distri, df)
y, yl, yu, y_middle = gen_outcomes(
    get_interval, cal_y_signal, beta, interval_bias, x, epsilon
)


x_eval, y_eval_signal = gen_eval(cal_y_signal, K, beta, n_test_points, x)

weights = np.random.beta(1, 1, [M, N])
yd = weights * yl + (1 - weights) * yu

params_dict = {
    "n": N,
    "M": M,
    "k": K,
    "beta": beta,
    "var_epsilon": var_epsilon,
    "interval_bias": interval_bias,
    "x_distri": x_distri,
    "epsilon_distri": epsilon_distri,
    "df": df,
    "tolerance": tolerance,
    "n_test_points": n_test_points,
    "x_eval": x_eval,
    "y_eval_signal": y_eval_signal,
}


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
# Plot the empirical distribution of y
plt.figure()
plt.hist([y, yd[0], yl], bins=100)
plt.show()
plt.figure()
plt.plot(y, yd[0], ".")
plt.show()

print(f"d_wasserstein(y,yd[0]) is {wasserstein_distance(y, yd[0])}")
print(f"d_wasserstein(y,y_middle) is {wasserstein_distance(y, y_middle)}")
# %%


model_linear = LinearRegression().fit(x_train, yd_train)
y_test_pred_linear = model_linear.predict(x_test).T
indices_linear = select_indices(
    compute_score, tolerance, yu_test, yl_test, y_test_pred_linear
)
y_eval_pred_linear = model_linear.predict(x_eval).T
y_pred_middle_linear = LinearRegression().fit(x, y_middle).predict(x_eval)
y_linear = LinearRegression().fit(x, y_middle).predict(x_eval)

plot_result(
    get_interval,
    **params_dict,
    indices=indices_linear,
    y_eval_pred=y_eval_pred_linear,
    y_mid_fit=y_pred_middle_linear,
    filename=f"{current_time()}-linear-{N}-{M}-{K}.pdf",
)
# %%


model_loclin = [LocLin(x_train, j) for j in yd_train.T]
y_test_pred_loclin = get_loclin_pred(x_test, model_loclin)
indices_loclin = select_indices(
    compute_score, tolerance, yu_test, yl_test, y_test_pred_loclin
)
y_eval_pred_loclin = get_loclin_pred(x_eval, model_loclin)
y_pred_middle_loclin = LocLin(x, y_middle).vec_fit(x_eval)

plot_result(
    get_interval,
    **params_dict,
    indices=indices_loclin,
    y_eval_pred=y_eval_pred_loclin,
    y_mid_fit=y_pred_middle_loclin,
    filename=f"{current_time()}-loclin-{N}-{M}-{K}.pdf",
)

# %%


# Create the Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=20)
rf.fit(x_train, yd_train)

# Make predictions
y_test_pred_rf = rf.predict(x_test).T
y_eval_pred_rf = rf.predict(x_eval).T
indices_rf = select_indices(compute_score, tolerance, yu_test, yl_test, y_test_pred_rf)
plot_result(
    get_interval,
    **params_dict,
    indices=indices_rf,
    y_eval_pred=y_eval_pred_rf,
    y_mid_fit=None,
    filename=f"{current_time()}-rf-{N}-{M}-{K}.pdf",
)

# %%

y = x[:, -1] * 1
yl, yu = get_interval(y, interval_bias)
y_middle = (yl + yu) / 2
yd = np.random.uniform(yl, yu, (M, N))
#
plt.figure()
plt.plot(x[:, -1], yd[0, :].T, ".", label="sample 1")
plt.plot(x[:, -1], yd[1, :].T, ".", label="sample 2")
plt.legend()
plt.title("Two drawed samples by uniformly drawing from the interval [y_l,y_u]")
plt.savefig(f"{current_time()}-sample-uniform.pdf", bbox_inches="tight")
plt.show()


#
weights = np.linspace(0, 1, 5)
yd = np.outer(weights, yl) + np.outer(1 - weights, yu)
plt.figure()
plt.plot(x[:, -1], yd[0, :].T, ".", label="sample 1")
plt.plot(x[:, -1], yd[1, :].T, ".", label="sample 2")
plt.legend()
plt.title(
    "Two drawed samples by first draw lambda_1, lambda_2 \n and form linear combination of y_l and y_u"
)
plt.savefig(f"{current_time()}-sample-fixed-weight.pdf", bbox_inches="tight")
plt.show()

# %%
