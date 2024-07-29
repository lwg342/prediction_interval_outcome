# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from utils.cross_validation import cv_bandwidth
from utils.empirical import visualize_prediction, analyze_and_plot


df = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")
df["Log_upper_bound"].describe()

# %%

alpha = 0.5
dt, result_all = analyze_and_plot(
    df,
    alpha=alpha,
)

dt_exact, result_exact = analyze_and_plot(
    df.loc[df["I_ERNVAL"] == 0], alpha=alpha, conformal_method="local"
)

dt_imputed, result_imputed = analyze_and_plot(
    df,
    yl_col="Log_income_with_imputed_lower",
    yu_col="Log_income_with_imputed_upper",
    alpha=alpha,
)
# %%

transform = np.array


plt.figure()

visualize_prediction(
    result_all,
    "prediction interval",
    transform,
    f"Prediction with exact number and range data",
    "tab:blue",
    offset=-0.1,
)
visualize_prediction(
    result_all,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number and range data",
    "tab:red",
    offset=-0.2,
)

visualize_prediction(
    result_exact,
    "conformal prediction interval",
    transform,
    f"Conformal prediction with exact number data",
    "tab:green",
    offset=0,
    marker="s",
)

visualize_prediction(
    result_imputed,
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
    f"Conformal Prediction Intervals when Experience is fixed at {result_all['experience']}"
)

plt.plot(
    result_all["edu"],
    result_all["kernel regression yl"],
    label="Mean of lower bound",
    marker="x",
)
plt.plot(
    result_all["edu"],
    result_all["kernel regression yu"],
    label="Mean of upper bound",
    marker="x",
)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {result_all['experience']}"
)
plt.savefig("conformal_intervals_empirical_edu.pdf")
