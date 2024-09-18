# %%
import dis
import numpy as np


def f(x):
    return (x - 1) ** 2 * (x + 1)


def g(x):
    return 4 * np.sqrt((x + 0.5) * (x >= -0.5))


def sigma2(x):
    return 0.25 + np.abs(x)  # You can replace this with any other function if required


def gen_x(n):
    return np.random.uniform(-1.5, 1.5, size=[n, 1])


def gen_y(X):
    Y = []
    for x in X:
        if np.random.rand() < 0.5:
            Y.append(np.random.normal(f(x) - g(x), np.sqrt(sigma2(x))))
        else:
            Y.append(np.random.normal(f(x) + g(x), np.sqrt(sigma2(x))))
    return np.array(Y).flatten()


def interval_censor(y, scale, **kwargs):
    # err1 = np.abs(np.random.normal(0, scale, y.shape))
    # err2 = np.abs(np.random.normal(0, scale, y.shape))
    err1 = np.random.exponential(0.1, size=y.shape)/2
    err2 = np.random.exponential(2.0, size=y.shape)/2
    # err1 =0
    # err2 = 0.2
    return (y - err1, y + err2)


def display_intervals(x_eval, interval, label="Prediction interval"):
    plt.figure(figsize=(10, 6))
    if interval is not None:
        for j in range(interval.shape[0]):
            if j == 0:
                plt.plot(
                    x_eval.flatten(),
                    interval[j],
                    color="tab:blue",
                    label="Prediction interval",
                )
            else:
                plt.plot(x_eval.flatten(), interval[j], color="tab:blue")

    plt.plot(data.x, data.yl, ".", color="tab:orange", markersize=4)
    plt.plot(data.x, data.yu, ".", color="tab:orange", markersize=4)
    plt.vlines(
        data.x,
        data.yl,
        data.yu,
        colors="tab:green",
        linestyles="solid",
        alpha=0.3,
        label="Interval (yl,yu)",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Intervals against x_eval")
    plt.legend()

    plt.tight_layout()
    plt.show()


def interpolate_pred_set(x_new, x_eval, pred_set_eval):
    # interpolate the prediction set based on x_eval and pred_set for the pred_set at x_test
    # Make sure x_eval is sorted
    output_pred_set = np.zeros((pred_interval_eval.shape[0], len(x_new)))
    for i, x_i in enumerate(x_new):
        # find nearest x_eval above and below x_i
        idx = np.abs(x_eval - x_i).argmin()

        if x_eval[idx] < x_i:
            id1, id2 = idx, idx + 1
        else:
            id1, id2 = idx - 1, idx
        weight = (x_i - x_eval[id1]) / (x_eval[id2] - x_eval[id1])
        output_pred_set[:, i] = (1 - weight) * pred_set_eval[
            :, id1
        ] + weight * pred_set_eval[:, id2]
    return output_pred_set


# %%
# [-] Example usage

from utils.simulation import IntervalCensoredData
from utils import *
from utils.set_estimation import *
import matplotlib.pyplot as plt

N = 1000
x = gen_x(N)
y = gen_y(x)
yl, yu = interval_censor(y, 0.5)
data = IntervalCensoredData(x, y, yl, yu)
x, yl, yu = data.x, data.yl, data.yu

x_eval = np.linspace(-1.5, 1.5, 50)[:, np.newaxis]

weights_eval = normalize_weights(
    multivariate_epanechnikov_kernel(x_eval, data.x, h=0.1)
)
display_intervals(x_eval, None)
# %%
# [-] Example: estimating 1 and multiple pred. intervals


weights = weights_eval[0]
grid = create_grid(*eligible_t0t1(yl, yu, weights), n_grid=100)
K = 2
interval_arr = create_valid_interval_array(grid, grid)

optim_interval = find_optimal_interval(interval_arr, weights, yl, yu, alpha=0.1)
print(optim_interval)
print(find_optimal_set(interval_arr, weights, yl, yu, alpha=0.1))

# %%
# [-] Example: estimating pred. set over x_eval
K = 1
bandwidth = 0.25
n_grid = 200
alpha = 0.1
rslt = np.zeros((2 * K, len(x_eval)))
weights_eval = normalize_weights(
    multivariate_epanechnikov_kernel(x_eval, data.x, bandwidth)
)
for i, x_e in enumerate(x_eval):
    weights = weights_eval[i]
    grid = create_grid(*eligible_t0t1(yl, yu, weights), n_grid=n_grid)
    interval_arr = create_valid_interval_array(grid, grid)
    if K == 1:
        optim_interval = find_optimal_interval(
            interval_arr, weights, yl, yu, alpha=alpha
        )
        rslt[:, i] = optim_interval
    if K == 2:
        optim_set = find_optimal_set(interval_arr, weights, yl, yu, alpha=alpha, K=2)
        rslt[:, i] = optim_set


# %%
# [-] Plot the intervals in rslt against x_eval

display_intervals(x_eval, rslt)

# %%
# [-] Conformal inference
pred_interval_eval = pred_interval(
    x_eval,
    data.x_train,
    data.yl_train,
    data.yu_train,
    h=bandwidth,
    option="two intervals",
)
pred_interval_test = interpolate_pred_set(data.x_test, x_eval, pred_interval_eval)
scores = np.nanmin(
    [
        np.maximum(
            np.sort(pred_interval_test, axis=0)[0] - data.yl_test,
            data.yu_test - np.sort(pred_interval_test, axis=0)[1],
        ),
        np.maximum(
            np.sort(pred_interval_test, axis=0)[2] - data.yl_test,
            data.yu_test - np.sort(pred_interval_test, axis=0)[3],
        ),
    ],
    axis=0,
)
scores
qq = np.quantile(scores, 1 - alpha)

conf_pred_eval = np.array(
    [
        pred_interval_eval[0] - qq,
        pred_interval_eval[1] + qq,
        pred_interval_eval[2] - qq,
        pred_interval_eval[3] + qq,
    ]
)
# %%


def display_intervals(x_eval, interval, label="Prediction interval"):
    plt.figure(figsize=(10, 6))
    if interval is not None:
        for j in range(interval.shape[0]):
            if j == 0:
                plt.plot(
                    x_eval.flatten(),
                    interval[j],
                    color="tab:blue",
                    label="Prediction interval",
                )
            else:
                plt.plot(x_eval.flatten(), interval[j], color="tab:blue")

    plt.plot(data.x, data.yl, ".", color="tab:orange", markersize=4)
    plt.plot(data.x, data.yu, ".", color="tab:orange", markersize=4)
    plt.vlines(
        data.x,
        data.yl,
        data.yu,
        colors="tab:green",
        linestyles="solid",
        alpha=0.3,
        label="Interval (yl,yu)",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Intervals against x_eval")
    plt.legend()

    plt.tight_layout()
    plt.show()


display_intervals(x_eval, pred_interval_eval)
display_intervals(x_eval, conf_pred_eval)

# %%
import statsmodels.api as sm


def quadratic_features(x):
    return np.hstack(
        [
            np.ones([x.shape[0], 1]),
            x,
            x**2,
        ]
    )


def cubic_features(x):
    return np.hstack(
        [
            np.ones([x.shape[0], 1]),
            x,
            x**2,
            x**3,
        ]
    )


construct_features = cubic_features

quantile_upper_model = sm.QuantReg(
    data.yl_train,
    construct_features(data.x_train),
).fit(q=alpha / 2)
quantile_lower_model = sm.QuantReg(
    data.yu_train,
    construct_features(data.x_train),
).fit(q=1 - alpha / 2)

# Predict intervals for test points
X_test = construct_features(data.x_test)
yl_pred_025 = quantile_upper_model.predict(X_test)
yu_pred_975 = quantile_lower_model.predict(X_test)
interval_quantile = np.array([yl_pred_025, yu_pred_975])
scores = np.maximum(
    interval_quantile[0] - data.yl_test, data.yu_test - interval_quantile[1]
)
qq = np.quantile(scores, [1 - alpha], method="higher")

pred_interval_eval_quantile = np.array(
    [
        quantile_upper_model.predict(construct_features(x_eval)),
        quantile_lower_model.predict(construct_features(x_eval)),
    ]
)
conformal_interval_eval_quantile = np.array(
    [pred_interval_eval_quantile[0] - qq, pred_interval_eval_quantile[1] + qq]
)
display_intervals(x_eval, pred_interval_eval_quantile)
# %%
vol_quantile = conformal_interval_eval_quantile[1] - conformal_interval_eval_quantile[0]


def replace_nan_with_mean(arr):
    arr1 = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        a = arr[:, i]
        a[np.isnan(a)] = np.mean(a[~np.isnan(a)])
        arr1[:, i] = a
    return arr1


conf_pred_eval_filled = replace_nan_with_mean(conf_pred_eval)
vol_cdf = (
    conf_pred_eval_filled[1]
    - conf_pred_eval_filled[0]
    + conf_pred_eval_filled[3]
    - conf_pred_eval_filled[2]
)

plt.plot(vol_cdf)
plt.plot(vol_quantile)


# %%
