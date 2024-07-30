# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils.cross_validation import cv_bandwidth
from utils.empirical import visualize_prediction, analyze_and_plot
import os

random_seed = 43
np.random.seed(random_seed)

filename = "apsp_jd23_eul_pwta22.tab"
df = pd.read_csv(f"wage-data/clean_{filename}")

df["Log_upper_bound"].describe()

# %%

alpha = 0.25
dt, results = analyze_and_plot(
    df,
    alpha=alpha,
    conformal_method="local",
)

dt_exact, results_exact = analyze_and_plot(
    df.loc[~df["range_indicator"]], alpha=alpha, conformal_method="local"
)
# %%

transform = np.array


plt.figure()

visualize_prediction(
    results,
    "prediction interval",
    transform,
    f"Prediction with exact number and range data",
    "tab:blue",
    offset=-0.3,
)
visualize_prediction(
    results,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number and range data",
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
    f"Prediction with exact number data",
    "tab:orange",
    offset=0.1,
)
visualize_prediction(
    results_exact,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number data",
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
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {results['experience']}"
)
plt.savefig("conformal_intervals_empirical_edu.pdf")
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
        "Kernel Regression Lower": results["kernel regression yl"],
        "Kernel Regression Upper": results["kernel regression yu"],
        "Random Seed": [results["random seed"]] * len(results["edu"]),
    }
)


# File path for the CSV

result_file_path = "UKDA_9248_results.csv"

if not os.path.isfile(result_file_path):
    df_results.to_csv(result_file_path, mode="w", header=True, index=False)
else:
    df_results.to_csv(result_file_path, mode="a", header=False, index=False)
df_results.to_csv(result_file_path, index=False, header=False, mode="a")

# %%
