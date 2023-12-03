# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wlpy.regression import LocLin
from wlpy.gist import current_time
from utils import *


# %%


def cal_y_signal(eval, params):
    return np.sqrt(2) + np.dot(eval, params)  # + 0.2 * eval[:, -1] ** 2
    # + np.cos(eval[:, -1])


n = 1000  # sample size
M = 100  # number of draws from intervals
k = 4  # number of covariates
beta = np.ones(k) * np.pi / 10
var_epsilon = 1.0
interval_bias = [0.0, 0.0]
x_distri = "uniform"
epsilon_distri = "chisquare"
df = 2
params_dict = create_param_dict(
    n, M, k, beta, var_epsilon, interval_bias, x_distri, epsilon_distri, df
)

x = gen_x(n, k, x_distri)
epsilon = gen_noise(n, var_epsilon, epsilon_distri, df)
y, yl, yu, y_middle = gen_outcomes(
    get_interval, cal_y_signal, beta, interval_bias, x, epsilon
)

weights = np.random.beta(1, 1, [M, n])
yd = weights * yl + (1 - weights) * yu

# %%
# Plot the empirical distribution of y
plt.figure()
plt.hist(y, bins=100, density=True)
plt.hist(yl, bins=100, density=True)
plt.hist(yd[0], bins=100, density=True)
plt.show()
plt.figure()
plt.plot(y, yd[0], ".")
plt.show()
# %%
from scipy.stats import wasserstein_distance

dist = wasserstein_distance(y, yd[0])
print(f"The Wasserstein distance between the two samples is {dist}")
# %%


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
tolerance = 1 / np.sqrt(n)

model_linear = LinearRegression().fit(x_train, yd_train)
# TODO: might want to parallel compute this part.
# model_kernel = [KernelReg(j, x_train, var_type=f"{'c'*k}") for j in yd_train]

y_test_pred_linear = model_linear.predict(x_test).T

indices_linear = select_indices(
    compute_score, tolerance, yu_test, yl_test, y_test_pred_linear
)
# print(f"number of indices slected is {indices.shape[0]}")
# %%
n_test_points = 100
evaluation_points = np.column_stack(
    (
        np.ones((n_test_points, k - 1)) * 0,
        np.linspace(np.min(x[:, -1]), np.max(x[:, -1]), n_test_points),
    )
)
y_signal = cal_y_signal(evaluation_points, beta)

y_eval_pred_linear = model_linear.predict(evaluation_points).T

y_pred_middle_linear = LinearRegression().fit(x, y_middle).predict(evaluation_points)

plot_result(
    get_interval,
    **params_dict,
    indices=indices_linear,
    evaluation_points=evaluation_points,
    y_signal=y_signal,
    y_eval_pred=y_eval_pred_linear,
    y_pred_middle=y_pred_middle_linear,
    filename=f"{current_time()}-linear-{n}-{M}-{k}.pdf",
)
# %%


model_loclin = [LocLin(x_train, j) for j in yd_train.T]
y_test_pred_loclin = get_loclin_pred(x_test, model_loclin)
indices_loclin = select_indices(
    compute_score, tolerance, yu_test, yl_test, y_test_pred_loclin
)
y_eval_pred_loclin = get_loclin_pred(evaluation_points, model_loclin)
y_pred_middle_loclin = LocLin(x, y_middle).vec_fit(evaluation_points)

plot_result(
    get_interval,
    n,
    M,
    k,
    interval_bias,
    indices_loclin,
    evaluation_points,
    y_signal,
    y_eval_pred_loclin,
    y_pred_middle_loclin,
    filename=f"{current_time()}-loclin-{n}-{M}-{k}.pdf",
)


# %%

y = x[:, -1] * 1
yl, yu = get_interval(y, interval_bias)
y_middle = (yl + yu) / 2
yd = np.random.uniform(yl, yu, (M, n))
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
