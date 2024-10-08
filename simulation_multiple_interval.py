# %%
import numpy as np
from utils.simulation import IntervalCensoredData
from utils import *
from utils.set_estimation import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tqdm
import pandas as pd
import os
from datetime import datetime


def f(x):
    return (x - 1) ** 2 * (x + 1)


def g(x):
    return 3 * np.sqrt((x + 0.5) * (x >= -0.5))


def sigma2(x):
    return 0.25 + np.abs(x)


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


def interval_censor(y, scale=0.5, **kwargs):
    # err1 = np.abs(np.random.normal(0, scale, y.shape))
    # err2 = np.abs(np.random.normal(0, scale, y.shape))
    # err1 = 0
    # err2 = 0
    # err1 = np.random.exponential(0.1, size=y.shape) / 2
    # err2 = np.random.exponential(2.0, size=y.shape) / 2
    # err1 =0
    # err2 = 0.2
    # return (y - err1, y + err2)

    rr = np.random.rand(y.shape[0])
    yl, yu = np.zeros_like(y), np.zeros_like(y)
    mask = rr < 0.1
    yl[mask], yu[mask] = y[mask], y[mask]

    yl[~mask] = np.floor(y[~mask] / scale) * scale
    yu[~mask] = (np.floor(y[~mask] / scale) + 1) * scale
    return (yl, yu)


def generate_data(sample_size=1000, scale=1):
    x = gen_x(sample_size)
    y = gen_y(x)
    yl, yu = interval_censor(y, scale=scale)
    data = IntervalCensoredData(x, y, yl, yu)
    return data


def generate_data_truth(x_eval, sample_size=1000):
    # generate sample_size data points for each x in x_eval
    X = x_eval @ np.ones([1, sample_size])
    y = np.zeros([x_eval.shape[0], sample_size])
    yl = np.zeros([x_eval.shape[0], sample_size])
    yu = np.zeros([x_eval.shape[0], sample_size])
    for i, x in enumerate(X):
        y[i] = gen_y(x)
        yl[i], yu[i] = interval_censor(y[i])
    return yl, yu


def interpolate_pred_set(x_new, x_eval, pred_set_eval):
    # interpolate the prediction set based on x_eval and pred_set for the pred_set at x_test
    # Make sure x_eval is sorted
    output_pred_set = np.zeros((pred_set_eval.shape[0], len(x_new)))
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


def conformity_scores(pred_interval, yl, yu, K=1):
    if K == 1:
        scores = np.maximum(pred_interval[0] - yl, yu - pred_interval[1])
    if K == 2:
        scores = np.nanmin(
            [
                np.maximum(
                    np.sort(pred_interval, axis=0)[0] - yl,
                    yu - np.sort(pred_interval, axis=0)[1],
                ),
                np.maximum(
                    np.sort(pred_interval, axis=0)[2] - yl,
                    yu - np.sort(pred_interval, axis=0)[3],
                ),
            ],
            axis=0,
        )

    return scores


def conformalise_interval(pred_test, pred_eval, data, alpha):
    scores = conformity_scores(pred_test, data.yl_test, data.yu_test, K=1)
    qq = np.quantile(
        scores, [(1 - alpha) * (1 + 1 / pred_test.shape[1])], method="higher"
    )

    conf_pred_interval = np.array([pred_eval[0] - qq, pred_eval[1] + qq])

    return conf_pred_interval


def conformalise_set(pred_test, pred_eval, yl_test, yu_test, alpha):
    scores = conformity_scores(pred_test, yl_test, yu_test, K=2)
    qq = np.quantile(scores, (1 - alpha) * (1 + 1 / pred_test.shape[1]))
    conf_pred_eval = np.array(
        [
            pred_eval[0] - qq,
            pred_eval[1] + qq,
            pred_eval[2] - qq,
            pred_eval[3] + qq,
        ]
    )

    return qq, conf_pred_eval


def quantile_models(data, alpha, construct_features):

    quantile_upper_model = sm.QuantReg(
        data.yl_train,
        construct_features(data.x_train),
    ).fit(q=alpha / 2)
    quantile_lower_model = sm.QuantReg(
        data.yu_train,
        construct_features(data.x_train),
    ).fit(q=1 - alpha / 2)

    return quantile_upper_model, quantile_lower_model


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


def quantile_pred(x, quantile_upper_model, quantile_lower_model, construct_features):
    X = construct_features(x)
    yl_pred = quantile_upper_model.predict(X)
    yu_pred = quantile_lower_model.predict(X)
    return np.array([yl_pred, yu_pred])


def conf_pred_quantile_method(alpha, data, x_eval, construct_features=cubic_features):
    quantile_upper_model, quantile_lower_model = quantile_models(
        data, alpha, construct_features
    )
    pred_intval_quantile_test = quantile_pred(
        data.x_test, quantile_upper_model, quantile_lower_model, construct_features
    )
    pred_intval_quantile_eval = quantile_pred(
        x_eval, quantile_upper_model, quantile_lower_model, construct_features
    )
    conf_pred_quantile_eval = conformalise_interval(
        pred_intval_quantile_test, pred_intval_quantile_eval, data, alpha=alpha
    )

    return conf_pred_quantile_eval


def compute_vol(pred_set, K=1):
    if K == 1:
        return pred_set[1] - pred_set[0]
    if K == 2:
        pred_set_filled = replace_nan_with_mean(pred_set)
        return (
            pred_set_filled[1]
            - pred_set_filled[0]
            + pred_set_filled[3]
            - pred_set_filled[2]
        )


def replace_nan_with_mean(arr):
    arr1 = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        a = arr[:, i].copy()
        a[np.isnan(a)] = np.mean(a[~np.isnan(a)])
        arr1[:, i] = a
    return arr1


def conformal_intervals_mid_merge(conf_pred):
    mask = conf_pred[1] >= conf_pred[2]
    conf_pred[1, mask] = np.nan
    conf_pred[2, mask] = np.nan
    return conf_pred


def coverage_ratio(x_eval, prediction, yl, yu, K=1):
    coverage = np.zeros(x_eval.shape[0])
    for i in range(x_eval.shape[0]):
        coverage[i] = np.mean(conformity_scores(prediction[:, i], yl[i], yu[i], K) < 0)
    return coverage


def display_intervals(
    x_eval,
    interval,
    data,
    label="Prediction interval",
    savefig=False,
    title="prediction_intervals.pdf",
):
    plt.figure(figsize=(16, 12))
    if interval is not None:
        for j in range(interval.shape[0]):
            if j == 0:
                plt.plot(
                    x_eval.flatten(),
                    interval[j],
                    # color="tab:red",
                    label="Prediction interval",
                    linestyle="--",
                    linewidth=3,
                )
            else:
                plt.plot(
                    x_eval.flatten(),
                    interval[j],
                    # color="grey",
                    linestyle="--",
                    linewidth=3,
                )
    sparse = 1000
    plt.plot(
        data.x[:sparse],
        data.yl[:sparse],
        "o",
        color="salmon",
        markersize=4,
        label="Lower bracket yl",
    )
    plt.plot(
        data.x[:sparse],
        data.yu[:sparse],
        "x",
        color="tab:blue",
        markersize=4,
        label="Upper bracket yu",
    )
    plt.vlines(
        data.x[:sparse],
        data.yl[:sparse],
        data.yu[:sparse],
        colors="grey",
        alpha=0.5,
        linewidth=0.5,
        linestyle="--",
        label="Interval (yl,yu)",
    )
    # for i in range(sparse):
    #     if i == 0:
    #         arrow = FancyArrowPatch(
    #             (data.x.flatten()[i], data.yl[i]),
    #             (data.x.flatten()[i], data.yu[i]),
    #             arrowstyle="|-|",
    #             linestyle="--",
    #             alpha=0.5,
    #             mutation_scale=2,
    #             label="Interval (yl,yu)",
    #         )
    #     else:
    #         arrow = FancyArrowPatch(
    #             (data.x.flatten()[i], data.yl[i]),
    #             (data.x.flatten()[i], data.yu[i]),
    #             arrowstyle="|-|",
    #             linestyle="--",
    #             alpha=0.5,
    #             mutation_scale=2,
    #         )

    #     plt.gca().add_patch(arrow)

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title("Intervals against x_eval")
    plt.legend(fontsize=20)

    plt.tight_layout()
    if savefig:
        plt.savefig(title)
    plt.show()


# display_intervals(
#     x_eval,
#     conf_pred_q_3,
#     data,
#     savefig=True,
#     title=f"prediction_intervals_q3.pdf",
# )
# %%
def execution(
    alpha=0.1, bandwidth=0.2, scale=1, iteration=0, plot=False, save_csv=False
):
    # [-] Generate data
    data = generate_data(sample_size=1000, scale=scale)
    n_eval = 50
    n_grid = 100
    n_partition = 10
    x_eval = np.linspace(-1.5, 1.5, n_eval)[:, np.newaxis]
    # [-]  Construct C_hat
    pred_eval = pred_interval(
        x_eval,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=bandwidth,
        alpha=alpha + 0.02,
        n_grid=n_grid,
        option="two intervals",
    )
    # [-] Construct C_tilde conformalised prediction set
    pred_test = interpolate_pred_set(data.x_test, x_eval, pred_eval)
    qq, conf_pred = conformalise_set(
        pred_test, pred_eval, data.yl_test, data.yu_test, alpha=alpha
    )

    conf_pred = conformal_intervals_mid_merge(conf_pred)
    # [-] Construct local conformalised prediction set

    partition = np.linspace(-1.5, 1.5, n_partition + 1)
    qq = np.zeros(n_partition)
    conf_pred_local = conf_pred.copy()
    x_test_in_partition = (np.digitize(data.x_test, partition) - 1).flatten()
    x_eval_in_partition = np.digitize(x_eval.flatten(), partition) - 1
    for j in range(n_partition):
        qq[j], _ = conformalise_set(
            pred_test[:, x_test_in_partition == j],
            pred_eval,
            data.yl_test[x_test_in_partition == j],
            data.yu_test[x_test_in_partition == j],
            alpha=alpha,
        )

        conf_pred_local[:, x_eval_in_partition == j] = np.array(
            [
                pred_eval[0, x_eval_in_partition == j] - qq[j],
                pred_eval[1, x_eval_in_partition == j] + qq[j],
                pred_eval[2, x_eval_in_partition == j] - qq[j],
                pred_eval[3, x_eval_in_partition == j] + qq[j],
            ]
        )
        conf_pred_local = conformal_intervals_mid_merge(conf_pred_local)
    # [-]  Construct C_quantile conformalised prediction set based on quantile regression
    conf_pred_q_3 = conf_pred_quantile_method(
        alpha, data, x_eval, construct_features=cubic_features
    )
    conf_pred_q_2 = conf_pred_quantile_method(
        alpha, data, x_eval, construct_features=quadratic_features
    )

    # [-] Compute the volume of the prediction set
    vol_cdf = compute_vol(conf_pred, K=2)
    vol_quantile_3 = compute_vol(conf_pred_q_3, K=1)
    vol_quantile_2 = compute_vol(conf_pred_q_2, K=1)
    vol_cdf_local = compute_vol(conf_pred_local, K=2)
    # [-] Compute the coverage
    # First generate true model
    yl_sample, yu_sample = generate_data_truth(x_eval, sample_size=5000)
    cov_cdf = coverage_ratio(x_eval, conf_pred, yl_sample, yu_sample, K=2)
    cov_cdf_local = coverage_ratio(x_eval, conf_pred_local, yl_sample, yu_sample, K=2)
    cov_quantile_3 = coverage_ratio(x_eval, conf_pred_q_3, yl_sample, yu_sample, K=1)
    cov_quantile_2 = coverage_ratio(x_eval, conf_pred_q_2, yl_sample, yu_sample, K=1)
    # [-] Save the results
    if save_csv:
        rslt_row = {
            "x_eval": x_eval.flatten(),
            "vol_cdf": vol_cdf,
            "vol_quantile_3": vol_quantile_3,
            "vol_quantile_2": vol_quantile_2,
            "cov_cdf": cov_cdf,
            "cov_quantile_3": cov_quantile_3,
            "cov_quantile_2": cov_quantile_2,
            "vol_cdf_local": vol_cdf_local,
            "cov_cdf_local": cov_cdf_local,
        }
        rslt = pd.DataFrame(rslt_row)
        rslt["alpha"] = alpha
        rslt["bandwidth"] = bandwidth
        rslt["iteration"] = iteration
        rslt["n_grid"] = n_grid
        if not os.path.isfile(output_file):
            rslt.to_csv(output_file, mode="w", header=True, index=False)
        else:
            rslt.to_csv(output_file, mode="a", header=False, index=False)

    # [-] Plots
    if plot:
        plt.figure()
        plt.plot(vol_cdf, label="CDF")
        plt.plot(vol_quantile_3, label="Quantile cubic")
        plt.plot(vol_quantile_2, label="Quantile quadratic")
        plt.plot(vol_cdf_local, label="Local")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(cov_cdf, label="CDF")
        plt.plot(cov_quantile_3, label="Quantile cubic")
        plt.plot(cov_quantile_2, label="Quantile quadratic")
        plt.plot(cov_cdf_local, label="Local")
        plt.legend()
        plt.show()
        display_intervals(
            x_eval,
            conf_pred_q_3,
            data,
            savefig=True,
            title=f"prediction_intervals_q3.pdf",
        )
        display_intervals(
            x_eval,
            conf_pred_q_2,
            data,
            savefig=True,
            title=f"prediction_intervals_q2.pdf",
        )
        display_intervals(
            x_eval, conf_pred, data, savefig=True, title=f"prediction_intervals.pdf"
        )
        display_intervals(
            x_eval,
            conf_pred_local,
            data,
            savefig=True,
            title=f"prediction_intervals_local.pdf",
        )
        return data, conf_pred, conf_pred_local, conf_pred_q_3, conf_pred_q_2


# data, conf_pred, conf_pred_local, conf_pred_q_3, conf_pred_q_2 = execution(
# alpha=0.1, bandwidth=0.25, scale=1, plot=True
# )
# %%
# [-] Simulation for nsim times
if __name__ == "__main__":
    alpha = 0.1
    bandwidth = 0.25
    current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"simulation_results_{alpha}_{current_date}_fixed_censoring.csv"

    nsim = 100
    for i in tqdm.tqdm(range(nsim)):
        execution(
            alpha=alpha,
            bandwidth=bandwidth,
            scale=1,
            iteration=i,
            plot=False,
            save_csv=True,
        )
# %%
# [-] Analyse the results
rslt_df = pd.read_csv("simulation_results_0.1_2024-09-20.csv")
# %%
df_avg = (
    rslt_df.groupby("x_eval")
    .agg(
        {
            "vol_cdf": "mean",
            "vol_cdf_local": "mean",
            "vol_quantile_3": "mean",
            "vol_quantile_2": "mean",
            "cov_cdf": "mean",
            "cov_cdf_local": "mean",
            "cov_quantile_3": "mean",
            "cov_quantile_2": "mean",
        }
    )
    .reset_index()
)
plt.figure()
df_avg.loc[(df_avg["x_eval"] > -1.45) & (df_avg["x_eval"] < 1.45)].plot(
    x="x_eval", y=["vol_cdf", "vol_cdf_local", "vol_quantile_3"]
)
plt.savefig(f"volume_simulation_{rslt_df['alpha'][0]}.pdf")
plt.show()
df_avg[["x_eval", "vol_cdf", "vol_cdf_local", "vol_quantile_3", "vol_quantile_2"]]
# %%
plt.figure()
df_avg.loc[(df_avg["x_eval"] > -1.45) & (df_avg["x_eval"] < 1.45)].plot(
    x="x_eval",
    y=["cov_cdf", "cov_cdf_local", "cov_quantile_3", "cov_quantile_2"],
    linestyle="--",
)
plt.hlines(1 - rslt_df["alpha"][0], -1.5, 1.5, color="black", linestyle="-")
plt.ylim([0.7, 1.0])
plt.savefig(f"coverage_simulation_{rslt_df['alpha'][0]}.pdf")
plt.show()
# %%
plt.figure()
vol_sum = rslt_df.groupby("iteration").agg(
    {
        "vol_cdf": "sum",
        "vol_cdf_local": "sum",
        "vol_quantile_2": "sum",
        "vol_quantile_3": "sum",
    }
)
vol_avg = vol_sum * 3 / 50
plt.figure()
plt.boxplot(
    vol_avg,
    labels=["Conf. pred.", "Local conf. pred.", "Quantile cubic", "Quantile quadratic"],
)
plt.savefig(f"volume_simulation_boxplot_{rslt_df['alpha'][0]}.pdf")
plt.show()
# %%
# [-] Example usage


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
display_intervals(x_eval, None, data)
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
bandwidth = 0.2
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


qq, conf_pred_eval = conformalise_set(pred_interval_eval, data, alpha)

display_intervals(x_eval, pred_interval_eval)
display_intervals(x_eval, conf_pred_eval)
print(f"Conformal correction is {qq}")
# %%
# Predict intervals for test points

construct_features = cubic_features
quantile_upper_model, quantile_lower_model = quantile_models(
    data, alpha, cubic_features
)

pred_intval_quantile_test = quantile_pred(
    data.x_test, quantile_upper_model, quantile_lower_model, construct_features
)


pred_interval_eval_quantile, conformal_interval_eval_quantile = conformalise_interval(
    data,
    x_eval,
    alpha,
    construct_features,
    quantile_upper_model,
    quantile_lower_model,
    pred_intval_quantile_test,
)
display_intervals(x_eval, pred_interval_eval_quantile)
# %%
vol_quantile = conformal_interval_eval_quantile[1] - conformal_interval_eval_quantile[0]


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
