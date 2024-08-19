# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils.cross_validation import cv_bandwidth
from utils.empirical import visualize_prediction, analyze_and_plot


current_date = pd.Timestamp.now().strftime("%Y%m%d")
df = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")
df["Log_upper_bound"].describe()

result_file_path = f"asec23pub_results_{current_date}.csv"
current_date = pd.Timestamp.now().strftime("%Y%m%d")
# %%
random_seed = 19260817
bandwidth = np.array([0.34287434, 2.00268184])
for alpha in np.array([0.1, 0.25, 0.5, 0.75, 0.9]):
    for exp_fixed in np.array([10, 15, 20, 25]):
        print(f"alpha: {alpha}, exp_fixed: {exp_fixed}")
        np.random.seed(random_seed)
        dt, results = analyze_and_plot(
            df,
            alpha=alpha,
            exp_fixed=exp_fixed,
            bandwidth=bandwidth,
            # [!] bandwidth = None for running cross validation, result in [0.34287434 2.00268184]
            conformal_method="local",
        )

        dt_exact, results_exact = analyze_and_plot(
            df.loc[df["I_ERNVAL"] == 0],
            alpha=alpha,
            exp_fixed=exp_fixed,
            bandwidth=bandwidth,
            conformal_method="local",
        )
        # dt_imputed, results_imputed = analyze_and_plot(
        #     df,
        #     yl_col="Log_income_with_imputed_lower",
        #     yu_col="Log_income_with_imputed_upper",
        #     alpha=alpha,
        # )
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
