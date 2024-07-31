# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils.cross_validation import cv_bandwidth
from utils.empirical import visualize_prediction, analyze_and_plot


df = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")
df["Log_upper_bound"].describe()

# %%
random_seed = 19260817
np.random.seed(random_seed)
alpha = 0.9
dt, results = analyze_and_plot(
    df,
    alpha=alpha,
)

dt_exact, results_exact = analyze_and_plot(
    df.loc[df["I_ERNVAL"] == 0], alpha=alpha, conformal_method="local"
)

dt_imputed, results_imputed = analyze_and_plot(
    df,
    yl_col="Log_income_with_imputed_lower",
    yu_col="Log_income_with_imputed_upper",
    alpha=alpha,
)

# %%
results["random seed"] = random_seed
df_results = pd.DataFrame(
    {
        "Education": results["edu"],
        "Experience": [results["experience"]] * len(results["edu"]),
        "Alpha": [results["alpha"]] * len(results["edu"]),
        "Prediction Lower Bound": results["prediction interval"][0],
        "Prediction Upper Bound": results["prediction interval"][1],
        "Conformal Prediction Lower Bound": results["conformal prediction interval"][0],
        "Conformal Prediction Upper Bound": results["conformal prediction interval"][1],
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
        "Prediction Lower Bound with Exact and Imputed Number": results_imputed[
            "prediction interval"
        ][0],
        "Prediction Upper Bound with Exact and Imputed Number": results_imputed[
            "prediction interval"
        ][1],
        "Conformal Prediction Lower Bound with Exact and Imputed Number": results_imputed[
            "conformal prediction interval"
        ][
            0
        ],
        "Conformal Prediction Upper Bound with Exact and Imputed Number": results_imputed[
            "conformal prediction interval"
        ][
            1
        ],
        "Conformal Correction": results_imputed["conformal correction"],
        "Kernel Regression Lower": results["kernel regression yl"],
        "Kernel Regression Upper": results["kernel regression yu"],
        "Random Seed": [results["random seed"]] * len(results["edu"]),
    }
)


# File path for the CSV

result_file_path = "ASEC_results.csv"

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
    f"Prediction with exact number and range data",
    "tab:blue",
    offset=-0.1,
)
visualize_prediction(
    results,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number and range data",
    "tab:red",
    offset=-0.2,
)

visualize_prediction(
    results_exact,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number data",
    "tab:green",
    offset=0,
    marker="s",
)

visualize_prediction(
    results_imputed,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number data and imputed data",
    "tab:purple",
    offset=0.2,
    marker="x",
)


plt.xlabel("Education")
plt.ylabel("Predicted log earnings")
plt.legend()
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {results['experience']}"
)

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
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {results['experience']}"
)

cdt = pd.to_datetime("today").strftime("%Y%m%d%H%M%S")
plt.savefig(f"empirical_asec_{alpha}_{cdt}.pdf")
# %%
