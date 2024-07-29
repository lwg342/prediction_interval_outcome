# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils.cross_validation import cv_bandwidth
from utils.empirical import visualize_prediction, analyze_and_plot

np.random.seed(42)

filename = "apsp_jd23_eul_pwta22.tab"
df = pd.read_csv(f"wage-data/clean_{filename}")

df["Log_upper_bound"].describe()

# %%

alpha = 0.1
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
    offset=-0.4,
)
visualize_prediction(
    results,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number and range data",
    "tab:red",
    offset=-0.2,
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
)
visualize_prediction(
    results_exact,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number data",
    "tab:green",
    offset=0.2,
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
# Save the results in a pickle file

# %%
print(np.exp(results["kernel regression yl"]))
print(np.exp(results["kernel regression yu"]))

# %%
