# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils.cross_validation import cv_bandwidth
from utils.empirical import visualize_prediction, analyze_and_plot
import os

current_date = pd.Timestamp.now().strftime("%Y%m%d")

filename = "apsp_jd23_eul_pwta22.tab"
df = pd.read_csv(f"wage-data/clean_{filename}")

result_file_path = "UKDA_9248_results.csv"
df["Log_upper_bound"].describe()

# %%
random_seed = 19260817
# alpha = 0.9
# exp_fixed = 15
bandwidth = np.array(
    [0.48453814, 2.3811055]
)  # This is from a cross-validation run before this implementation
for alpha in np.array([0.75]):
    # for alpha in np.array([0.5]):
    for exp_fixed in np.array([15]):
        # for exp_fixed in np.array([15, 30]):
        np.random.seed(random_seed)
        print(f"alpha: {alpha}, exp_fixed: {exp_fixed}")
        dt, results = analyze_and_plot(
            df,
            alpha=alpha,
            exp_fixed=exp_fixed,
            conformal_method="local", 
            bandwidth=bandwidth,
        )

        dt_exact, results_exact = analyze_and_plot(
            df.loc[~df["range_indicator"]],
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
