# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.empirical import visualize_prediction, analyze_and_plot
import os
from sklearn.model_selection import train_test_split


current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

data = pd.read_csv("wage-data/clean_apsp_jd23_eul_pwta22.tab")
df = data[~data["Is_holdout"]].copy()
df_holdout = data[data["Is_holdout"]].copy()

result_file_path = f"UKDA_9248_results_{current_date}.csv"
df["Log_upper_bound"].describe()

# %%

# alpha = 0.9
# exp_fixed = 15
bandwidth = np.array(
    [0.48453814, 2.3811055]
)  # This is from a cross-validation run before this implementation
for random_seed in np.arange(20) + 43:
    for alpha in np.array([0.1, 0.5, 0.9]):
        for exp_fixed in np.array([10, 20]):
            np.random.seed(random_seed)
            print(f"alpha: {alpha}, exp_fixed: {exp_fixed}")
            dt, results = analyze_and_plot(
                df,
                alpha=alpha,
                exp_fixed=exp_fixed,
                conformal_method="local",
                bandwidth=bandwidth,
            )
            np.random.seed(random_seed)
            dt_exact, results_exact = analyze_and_plot(
                df.loc[~df["Range_report"]],
                alpha=alpha,
                exp_fixed=exp_fixed,
                conformal_method="local",
                bandwidth=bandwidth,
            )

            results["random seed"] = random_seed
            df_results = pd.DataFrame(
                {
                    "Education": results["edu"],
                    "Experience": [results["experience"]] * len(results["edu"]),
                    "Alpha": [results["alpha"]] * len(results["edu"]),
                    "Prediction Lower Bound": results["prediction interval"][0],
                    "Prediction Upper Bound": results["prediction interval"][1],
                    "Conformal Prediction Lower Bound": results[
                        "conformal prediction interval"
                    ][0],
                    "Conformal Prediction Upper Bound": results[
                        "conformal prediction interval"
                    ][1],
                    "Conformal Correction": results["conformal correction"],
                    "Prediction Lower Bound with Exact Number": results_exact[
                        "prediction interval"
                    ][0],
                    "Prediction Upper Bound with Exact Number": results_exact[
                        "prediction interval"
                    ][1],
                    "Conformal Prediction Lower Bound with Exact Number": results_exact[
                        "conformal prediction interval"
                    ][0],
                    "Conformal Prediction Upper Bound with Exact Number": results_exact[
                        "conformal prediction interval"
                    ][1],
                    "Conformal Correction": results_exact["conformal correction"],
                    "Kernel Regression Lower": results["kernel regression yl"],
                    "Kernel Regression Upper": results["kernel regression yu"],
                    "Random Seed": [results["random seed"]] * len(results["edu"]),
                    "Date": [current_date] * len(results["edu"]),
                    "Experience Bandwidth": [bandwidth[1]] * len(results["edu"]),
                }
            )

            # File path for the CSV

            if not os.path.isfile(result_file_path):
                df_results.to_csv(result_file_path, mode="w", header=True, index=False)
            else:
                df_results.to_csv(result_file_path, mode="a", header=False, index=False)


# %%
# [-] Check for why some prediction results are the same
random_seed = 19260817
# alpha = 0.9
# exp_fixed = 15
bandwidth = np.array(
    [0.48453814, 2.3811055]
)  # This is from a cross-validation run before this implementation
for alpha in np.array([0.75]):
    # for alpha in np.array([0.5]):
    for exp_fixed in np.array([20]):
        # for exp_fixed in np.array([15, 30]):
        np.random.seed(random_seed)
        print(f"alpha: {alpha}, exp_fixed: {exp_fixed}")
        dt, results = analyze_and_plot(
            df,
            alpha=alpha,
            exp_fixed=exp_fixed,
            conformal_method="none",
            bandwidth=bandwidth,
        )
# %%
# [-] Hold-out dataset coverage


def local_cond(df, edu, exp_l, exp_u):
    cond = (
        (df["Education"] == edu)
        & (df["Experience"] >= exp_l)
        & (df["Experience"] <= exp_u)
    )
    return cond


edu = 12
local_cond = local_cond(df_holdout, edu, 6, 14)
print(sum(local_cond) * 0.9)
print(
    sum(
        (
            df_holdout[local_cond].Log_lower_bound.to_numpy()
            >= df_results.loc[
                df_results["Education"] == edu, "Conformal Prediction Lower Bound"
            ].to_numpy()
        )
        & (
            df_holdout[local_cond].Log_upper_bound.to_numpy()
            <= df_results.loc[
                df_results["Education"] == edu, "Conformal Prediction Upper Bound"
            ].to_numpy()
        )
    )
)
print(
    sum(
        (
            df_holdout[local_cond].Log_lower_bound.to_numpy()
            >= df_results.loc[
                df_results["Education"] == edu,
                "Conformal Prediction Lower Bound with Exact Number",
            ].to_numpy()
        )
        & (
            df_holdout[local_cond].Log_upper_bound.to_numpy()
            <= df_results.loc[
                df_results["Education"] == edu,
                "Conformal Prediction Upper Bound with Exact Number",
            ].to_numpy()
        )
    )
)


# %%
# [-] Plots
import pandas as pd

transform = np.array


plt.figure()

visualize_prediction(
    results,
    "prediction interval",
    transform,
    f"Prediction (exact and range)",
    "tab:blue",
    offset=-0.3,
)
visualize_prediction(
    results,
    "conformal prediction interval",
    transform,
    f"Conf. prediction (exact and range)",
    "tab:red",
    offset=-0.1,
)
print(
    np.exp(results["conformal prediction interval"][0]),
    np.exp(results["conformal prediction interval"][1]),
)

visualize_prediction(
    results_exact,
    "prediction interval",
    transform,
    f"Prediction (exact only)",
    "tab:orange",
    marker="s",
    offset=0.1,
)
visualize_prediction(
    results_exact,
    "conformal prediction interval",
    transform,
    f"Conf. prediction (exact only)",
    "tab:green",
    offset=0.3,
    marker="s",
)
print(
    np.exp(results_exact["conformal prediction interval"][0]),
    np.exp(results_exact["conformal prediction interval"][1]),
)
plt.xlabel("Education")
plt.ylabel("Predicted log earnings")


plt.plot(
    results["edu"],
    results["kernel regression yl"],
    label="Mean of lower bound",
    marker="x",
)
plt.plot(
    results["edu"],
    results["kernel regression yu"],
    label="Mean of upper bound",
    marker="x",
)
# plt.legend(loc="center right", bbox_to_anchor=(-0.5, -0.1), ncol=1)
# Place the legend outside the plot area
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {results['experience']}"
)
cdt = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"empirical_edu_lfs_{alpha}_{cdt}.pdf", bbox_inches="tight")
