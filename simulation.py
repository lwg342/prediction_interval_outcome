# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm

from utils import *
from wlpy.gist import current_time
from utils.cross_validation import cv_bandwidth
from utils.simulation import SimData

cdt = current_time()
sns.set_style("ticks")
print(cdt)


# %%
def get_interval(y, scale, **kwargs):
    # err1 = np.random.chisquare(df=1, size=y.shape) * scale
    # err2 = np.random.chisquare(df=2, size=y.shape) * scale

    # err1 = np.random.exponential(0.5, size=y.shape)
    # err2 = np.random.exponential(0.5, size=y.shape)

    # err1 = np.random.exponential(0.5, size=y.shape)
    # err2 = np.random.exponential(2.0, size=y.shape)

    err1 = 0.0
    err2 = 0.0

    # err1 = np.random.uniform(0, 0.1, size=y.shape)
    # err2 = np.random.uniform(0, 4, size=y.shape)

    # err1 = np.abs(np.random.normal(0, scale * 0.5, size=y.shape))
    # err2 = np.abs(np.random.normal(0, scale * 10, size=y.shape))

    return (y - err1, y + err2)


dgp_params = {
    "N": 1000,
    "K": 1,
    "eps_std": 1.0,
    "pos": [0],
    "scale": 1.0,
}
gen_y_signal = lambda x, pos, **kwargs: np.inner(x[:, pos], np.ones(len(pos)))

nsim = 500
n_eval = 50


# %%
data_eval = SimData(
    get_interval=get_interval, gen_y_signal=gen_y_signal, dgp_params=dgp_params
)
data_eval.gen_eval(n_eval)
x_eval_fixed = data_eval.x_eval
y_new = data_eval.y_eval_samples
yl_new = data_eval.yl_eval_samples
yu_new = data_eval.yu_eval_samples

oracle_interval = np.zeros([2, data_eval.x_eval.shape[0]])
sample_size = 100000
data_eval.gen_eval(n_eval=n_eval, sample_size=sample_size)
eps_pm_kappa = np.array(
    data_eval.get_interval(data_eval.eps_eval_samples[:, 0], **dgp_params)
).T
oracle_interval_0 = find_oracle_interval(
    eps_pm_kappa,
    n=sample_size,
    alpha=0.05,
)
oracle_interval[0] = oracle_interval_0[0] + data_eval.y_eval_signal
oracle_interval[1] = oracle_interval_0[1] + data_eval.y_eval_signal
print(oracle_interval_0)
print(np.quantile(eps_pm_kappa[:, 0], 0.025), np.quantile(eps_pm_kappa[:, 1], 0.975))
# %%
empirical_prob_rslt_cdf = np.zeros([nsim, n_eval])
empirical_prob_rslt_cdf_before_conform = np.zeros([nsim, n_eval])
empirical_prob_rslt_quantile = np.zeros([nsim, n_eval])
empirical_prob_rslt_cdf_interval = np.zeros([nsim, n_eval])
empirical_prob_rslt_cdf_interval_before_conform = np.zeros([nsim, n_eval])
empirical_prob_rslt_cdf_interval_quantile = np.zeros([nsim, n_eval])
interval_width_rslt_cdf = np.zeros([nsim, n_eval])
interval_width_rslt_cdf_before_conform = np.zeros([nsim, n_eval])
interval_width_rslt_quantile = np.zeros([nsim, n_eval])

candidate_bandwidth = 0.3 * np.arange(1, 10) * silvermans_rule(data_eval.x_train)

h_cv, coverage_results, _ = cv_bandwidth(data_eval, candidate_bandwidth, alpha=0.05)
print(f"Cross validated bandwidth: {h_cv}")
mse = np.mean((coverage_results - (1 - 0.05)) ** 2, axis=1)
print(f"Mean squared error: {mse}")
plt.plot(candidate_bandwidth, mse)
# h_cv = 0.47
# %%
for j in tqdm(range(nsim)):

    data = SimData(
        get_interval=get_interval, gen_y_signal=gen_y_signal, dgp_params=dgp_params
    )

    candidate_bandwidth = 0.3 * np.arange(1, 10) * silvermans_rule(data.x_train)

    h_cv, coverage_results, _ = cv_bandwidth(data, candidate_bandwidth, alpha=0.05)
    pred_interval_test = pred_interval(
        data.x_test,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=h_cv,
    )
    scores = np.maximum(
        pred_interval_test[0] - data.yl_test, data.yu_test - pred_interval_test[1]
    )
    qq = np.quantile(scores, [0.95], method="higher")
    # print(qq)

    pred_interval_eval = pred_interval(
        x_eval_fixed,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=h_cv,
    )
    conformal_interval_eval = np.array(
        [pred_interval_eval[0] - qq, pred_interval_eval[1]] + qq
    )

    quantile_025_model = sm.QuantReg(data.yl_train, sm.add_constant(data.x_train)).fit(
        q=0.025
    )
    quantile_975_model = sm.QuantReg(data.yu_train, sm.add_constant(data.x_train)).fit(
        q=0.975
    )

    # Predict intervals for test points
    X_test = sm.add_constant(data.x_test)
    yl_pred_025 = quantile_025_model.predict(X_test)
    yu_pred_975 = quantile_975_model.predict(X_test)
    interval_quantile = np.array([yl_pred_025, yu_pred_975])
    scores = np.maximum(
        interval_quantile[0] - data.yl_test, data.yu_test - interval_quantile[1]
    )
    qq = np.quantile(scores, [0.95], method="higher")
    pred_interval_eval_quantile = np.array(
        [
            quantile_025_model.predict(sm.add_constant(x_eval_fixed)),
            quantile_975_model.predict(sm.add_constant(x_eval_fixed)),
        ]
    )
    conformal_interval_eval_quantile = np.array(
        [pred_interval_eval_quantile[0] - qq, pred_interval_eval_quantile[1]] + qq
    )

    empirical_prob_rslt_cdf[j] = calculate_proportion(y_new, conformal_interval_eval)
    empirical_prob_rslt_cdf_before_conform[j] = calculate_proportion(
        y_new, pred_interval_eval
    )
    empirical_prob_rslt_quantile[j] = calculate_proportion(
        y_new, conformal_interval_eval_quantile
    )

    empirical_prob_rslt_cdf_interval[j] = calculate_proportion_interval(
        yl_new, yu_new, conformal_interval_eval
    )
    empirical_prob_rslt_cdf_interval_before_conform[j] = calculate_proportion_interval(
        yl_new, yu_new, pred_interval_eval
    )
    empirical_prob_rslt_cdf_interval_quantile[j] = calculate_proportion_interval(
        yl_new, yu_new, conformal_interval_eval_quantile
    )

    interval_width_rslt_cdf[j] = conformal_interval_eval[1] - conformal_interval_eval[0]
    interval_width_rslt_cdf_before_conform[j] = (
        pred_interval_eval[1] - pred_interval_eval[0]
    )
    interval_width_rslt_quantile[j] = (
        conformal_interval_eval_quantile[1] - conformal_interval_eval_quantile[0]
    )
# %%
# Plot for coverage probability using the CDF method with adjusted marker properties
plt.figure(figsize=(10, 6))
plt.plot(
    x_eval_fixed,
    empirical_prob_rslt_cdf_interval.mean(axis=0),
    markersize=5,
    alpha=0.7,
    label="Coverage Probability of [YL, YU] (CDF Method)",
)
plt.plot(
    x_eval_fixed,
    empirical_prob_rslt_cdf_interval_before_conform.mean(axis=0),
    markersize=5,
    alpha=0.7,
    label="Coverage Probability of [YL, YU] (CDF Method) before conformalisation",
)

# Plot for coverage probability using the quantile method with adjustments
plt.plot(
    x_eval_fixed,
    empirical_prob_rslt_cdf_interval_quantile.mean(axis=0),
    markersize=5,  # Consistent marker size for uniformity
    alpha=0.7,  # Consistent opacity
    # color='crimson',  # Distinct and appealing color for differentiation
    label="Coverage Probability of [YL, YU] (Quantile Method)",
)
plt.plot(
    x_eval_fixed,
    calculate_proportion_interval(yl_new, yu_new, oracle_interval),
    # "o",  # Circle marker
    markersize=5,  # Consistent marker size for uniformity
    alpha=0.7,  # Consistent opacity
    # color='crimson',  # Distinct and appealing color for differentiation
    label="Coverage Probability of [YL, YU] (Oracle)",
)

plt.hlines(0.95, x_eval_fixed.min(), x_eval_fixed.max(), color="black", linestyle="--")
plt.plot(
    x_eval_fixed,
    empirical_prob_rslt_cdf.mean(axis=0),
    # "o",  # Circle marker
    markersize=5,  # Adjusted marker size for better visibility
    alpha=0.7,  # Slightly increased opacity
    label="Coverage Probability of Y (CDF Method)",
)

# Plot for coverage probability using the quantile method with adjustments
plt.plot(
    x_eval_fixed,
    empirical_prob_rslt_quantile.mean(axis=0),
    # "o",  # Circle marker
    markersize=5,  # Consistent marker size for uniformity
    alpha=0.7,  # Consistent opacity
    label="Coverage Probability of Y (Quantile Method)",
)

# Enhancing labels and adding a title for context
plt.ylabel("Empirical Probability", fontsize=12)
plt.xlabel("x_eval", fontsize=12)
plt.title("Empirical Coverage Probability Across x_eval", fontsize=12)

plt.legend(frameon=True, loc="best", fontsize=12)
plt.ylim(0.9, 1.0)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

plt.savefig(f"simulation-results/{cdt}-coverage.pdf", dpi=300)
plt.show()
# First scatter plot with increased marker size and adjusted alpha for better visibility
plt.figure(figsize=(10, 6))
plt.plot(
    x_eval_fixed,
    interval_width_rslt_cdf.mean(axis=0),
    alpha=0.7,
    label="Conformal Set (CDF)",
)
plt.plot(
    x_eval_fixed,
    interval_width_rslt_cdf_before_conform.mean(axis=0),
    alpha=0.7,
    label="Conformal Set (CDF) before conformalisation",
)

plt.plot(
    x_eval_fixed,
    interval_width_rslt_quantile.mean(axis=0),
    alpha=0.5,
    label="Conformal Set (quantile)",
)
plt.plot(
    x_eval_fixed,
    oracle_interval[1] - oracle_interval[0],
    alpha=0.5,
    label="Conformal Set (Oracle)",
)

# Enhancing the labels and title
plt.ylabel("Interval width", fontsize=12)
plt.xlabel("x_eval", fontsize=12)

plt.title("Conformal Interval Widths", fontsize=12)

# Adding the legend with a specified font size
plt.legend(fontsize=12, frameon=True, shadow=True, borderpad=1)

# Adding a grid for better readability (the style may already include this)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Saving the figure
plt.tight_layout()  # Adjust the padding between and around subplots.
plt.savefig(
    f"simulation-results/{cdt}-interval-width.pdf", dpi=300
)  # Specifying a high resolution
plt.show()
print(cdt)


# %%
# compute the symmetric differen between two intervals
def symmetric_difference(interval_1, interval_2):
    return np.abs(interval_1[0] - interval_2[0]) + np.abs(interval_1[1] - interval_2[1])


sym_diff= symmetric_difference(oracle_interval, pred_interval_eval)
plt.plot(sym_diff)
# %%
