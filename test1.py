# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wlpy.regression import LocLin
from wlpy.gist import current_time


def compute_score(y_test_prediction, yu_test, yl_test):
    diff_l = np.maximum(yl_test - y_test_prediction, 0) ** 2
    diff_u = np.maximum(y_test_prediction - yu_test, 0) ** 2
    return (diff_l + diff_u).mean(axis=1)


def get_interval(y, a1, a2):
    yl = np.floor(y) - a1
    yu = np.ceil(y) + a2
    return yl, yu


def get_loclin_pred(eval, loclin_models):
    pred = np.column_stack([mm.vec_fit(eval) for mm in loclin_models]).T
    return pred


def select_indices(compute_score, tolerance, yu_test, yl_test, y_test_pred):
    score = compute_score(y_test_pred, yu_test, yl_test)
    smallest_score = np.min(score)
    threshold = smallest_score + tolerance
    indices = np.where(score < threshold)[0]
    return indices


def plot_result(
    get_interval,
    n,
    a1,
    a2,
    indices,
    evaluation_points,
    y_signal,
    y_eval_pred,
    y_pred_middle=None,
    filename=None,
):
    plt.figure()

    plt.plot(evaluation_points[:, -1], y_signal, label="y_signal")
    yl_signal, yu_signal = get_interval(y_signal, a1, a2)
    plt.plot(evaluation_points[:, -1], yl_signal, label="y_interval", color="green")
    plt.plot(evaluation_points[:, -1], yu_signal, label="y_interval", color="green")

    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred.min(axis=0),
        color="red",
        label="pre-selection prediction",
    )
    plt.plot(evaluation_points[:, -1], y_eval_pred.max(axis=0), color="red")

    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred[indices].min(axis=0),
        "+",
        color="orange",
        label="post-section prediction",
    )
    plt.plot(
        evaluation_points[:, -1], y_eval_pred[indices].max(axis=0), "+", color="orange"
    )
    plt.xlabel("f$x_{k}$")
    if y_pred_middle is not None:
        plt.plot(
            evaluation_points[:, -1],
            y_pred_middle,
            label="fit y_middle",
            color="purple",
        )

    plt.title(
        f"$n$={n}, $M$={M}, $k$={k}, $v_\epsilon$={var_epsilon}, $a_1$={a1}, $a_2$={a2}, \n $x$_distri={x_distri}, $\epsilon$_distri={epsilon_distri}, df={df}"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


# %%


def cal_y_signal(eval, params):
    return np.sqrt(2) + np.dot(eval, params)# + 0.2 * eval[:, -1] ** 2
    # + np.cos(eval[:, -1])


n = 1000  # sample size
M = 100  # number of draws from intervals
k = 4  # number of covariates
# beta = np.linspace(1, k, k)* np.pi/10
beta = np.ones(k) * np.pi / 10
var_epsilon = 1.0
a1, a2 = 2.0, 0.0
x_distri = "normal"
epsilon_distri = "normal"
df = 1


if x_distri == "normal":
    x = np.random.normal(0, 1, (n, k))
if x_distri == "uniform":
    x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (n, k))

if epsilon_distri == "normal":
    epsilon = np.random.normal(0, var_epsilon, n)
if epsilon_distri == "chisquare":
    epsilon = (np.random.chisquare(df, n) - df) / np.sqrt(2 * df)
if epsilon_distri == "no_noise":
    epsilon = np.zeros(n)

y = cal_y_signal(x, beta) + epsilon
yl, yu = get_interval(y, a1, a2)
y_middle = (yl + yu) / 2

# Plot the empirical distribution of y conditional on it falls into one of the intervals
plt.figure()
plt.hist(y, bins=100, density=True)
plt.show()
# %%
# yd = np.random.uniform(yl, yu, (M, n))
weights = np.random.uniform(0, 1, M)
yd = np.outer(weights, yl) + np.outer(1 - weights, yu)

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
model_linear = LinearRegression().fit(x_train, yd_train)
# TODO: might want to parallel compute this part.
# model_kernel = [KernelReg(j, x_train, var_type=f"{'c'*k}") for j in yd_train]
model_loclin = [LocLin(x_train, j) for j in yd_train.T]

y_test_pred_linear = model_linear.predict(x_test).T
y_test_pred_loclin = get_loclin_pred(x_test, model_loclin)

tolerance = 1 / np.sqrt(n)
indices_linear = select_indices(
    compute_score, tolerance, yu_test, yl_test, y_test_pred_linear
)
indices_loclin = select_indices(
    compute_score, tolerance, yu_test, yl_test, y_test_pred_loclin
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
y_eval_pred_loclin = get_loclin_pred(evaluation_points, model_loclin)

y_pred_middle_linear = LinearRegression().fit(x, y_middle).predict(evaluation_points)
y_pred_middle_loclin = LocLin(x, y_middle).vec_fit(evaluation_points)

plot_result(
    get_interval,
    n,
    a1,
    a2,
    indices_linear,
    evaluation_points,
    y_signal,
    y_eval_pred_linear,
    y_pred_middle_linear,
    filename=f"{current_time()}-linear-{n}-{M}-{k}.pdf",
)
plot_result(
    get_interval,
    n,
    a1,
    a2,
    indices_loclin,
    evaluation_points,
    y_signal,
    y_eval_pred_loclin,
    y_pred_middle_loclin,
    filename=f"{current_time()}-loclin-{n}-{M}-{k}.pdf",
)


# %%
# yd = np.random.uniform(yl, yu, (M, n))
# weights = np.random.uniform(0, 1, M)
# yd = np.outer(weights, yl) + np.outer(1 - weights, yu)
# plt.plot(x[:, -1], yd[:5,:].T, 'o')
# 
# %%
