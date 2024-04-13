# %%
import numpy as np
from utils_sim import *
from wlpy.gist import current_time
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm

sns.set_style("ticks")

cdt = current_time()
print(cdt)


# %%
def get_interval(y, scale, **kwargs):
    # err1 = np.random.chisquare(df=1, size=y.shape) * scale
    # err2 = np.random.chisquare(df=2, size=y.shape) * scale

    # err1 = np.random.exponential(0.1, size=y.shape)
    # err2 = np.random.exponential(0.2, size=y.shape)

    # err1 = np.random.exponential(0.5, size=y.shape)
    # err2 = np.random.exponential(1.5, size=y.shape)

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

data = Data(get_interval=get_interval, gen_y_signal=gen_y_signal, dgp_params=dgp_params)
# %%

plt.scatter(data.x.flatten(), data.y, alpha=0.7)
plt.scatter(data.x.flatten(), data.yl, alpha=0.7)
plt.scatter(data.x.flatten(), data.yu, alpha=0.7)

weights = compute_weights(
    multivariate_epanechnikov_kernel(
        0,
        data.x,
        h=silvermans_rule(data.x),
    )
)

t0c, t1c = eligible_t0t1(data.yl, data.yu, weights)
print(t0c, t1c)

# %%
nsim = 100

empirical_prob_rslt_cdf = np.zeros([nsim, data.x_eval.shape[0]])
empirical_prob_rslt_cdf_before_conform = np.zeros([nsim, data.x_eval.shape[0]])
empirical_prob_rslt_quantile = np.zeros([nsim, data.x_eval.shape[0]])
empirical_prob_rslt_cdf_interval = np.zeros([nsim, data.x_eval.shape[0]])
empirical_prob_rslt_cdf_interval_before_conform = np.zeros([nsim, data.x_eval.shape[0]])
empirical_prob_rslt_cdf_interval_quantile = np.zeros([nsim, data.x_eval.shape[0]])
interval_width_rslt_cdf = np.zeros([nsim, data.x_eval.shape[0]])
interval_width_rslt_quantile = np.zeros([nsim, data.x_eval.shape[0]])

data_eval = Data(
    get_interval=get_interval, gen_y_signal=gen_y_signal, dgp_params=dgp_params
)
x_eval_fixed = data_eval.x_eval
y_new = data_eval.y_eval_samples
yl_new = data_eval.yl_eval_samples
yu_new = data_eval.yu_eval_samples

for j in tqdm(range(nsim)):
    data = Data(
        get_interval=get_interval, gen_y_signal=gen_y_signal, dgp_params=dgp_params
    )

    pred_interval_test = pred_interval(data.x_test, data)
    scores = np.maximum(
        pred_interval_test[0] - data.yl_test, data.yu_test - pred_interval_test[1]
    )
    qq = np.quantile(scores, [0.95], method="higher")
    print(qq)

    pred_interval_eval = pred_interval(x_eval_fixed, data)
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
    qq = np.quantile(scores, [0.95])
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
    interval_width_rslt_quantile[j] = (
        conformal_interval_eval_quantile[1] - conformal_interval_eval_quantile[0]
    )

# %%


def find_oracle_interval(intvs, n, alpha):
    sorted_intvs = intvs[np.argsort(intvs[:, 0])]
    i = 0
    optimal_width = np.inf
    while i <= n * alpha:
        # print(i)
        sorted_upper = np.sort(sorted_intvs[i:, 1])[::-1]
        # print(sorted_upper)

        t1 = sorted_upper[np.floor(n * alpha - i).astype(int)]

        width = t1 - sorted_intvs[i, 0]
        if width < optimal_width:
            optimal_width = width
            optimal_interval = [
                sorted_intvs[i, 0],
                t1,
            ]
        i += 1
    # print(optimal_interval)
    return optimal_interval


n = 100000
alpha = 0.05
ee = np.random.normal(loc=0, scale=1, size=(n))
intvs = np.array([ee, ee]).T


# Find oracle interval
oracle_interval = find_oracle_interval(intvs, n, alpha)
print(oracle_interval, oracle_interval[1] - oracle_interval[0])
# %%
oracle_interval = np.zeros([2, data.x_eval.shape[0]])
for k in range(data.x_eval.shape[0]):
    intvs = np.array([yl_new[:, k], yu_new[:, k]]).T
    oracle_interval[:, k] = find_oracle_interval(
        intvs, n=data.yl_eval_samples.shape[0], alpha=0.05
    )
    print(k)
plt.plot(x_eval_fixed, oracle_interval[1, :])
plt.plot(x_eval_fixed, oracle_interval[0, :])


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
    label="Coverage Probability of [YL, YU] (CDF Method)",
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
    label="Conformal Set (cdf)",
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
