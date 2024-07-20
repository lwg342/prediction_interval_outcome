# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from cross_validation import cv_bandwidth
from sklearn.kernel_ridge import KernelRidge


class EmpiricalData:
    def __init__(self, df):
        self.x = df[["Edu", "Exp"]].to_numpy()
        self.yl = df["Log_Weekly_Lower"].to_numpy()
        self.yu = df["Log_Weekly_Upper"].to_numpy()
        (
            self.x_train,
            self.x_test,
            self.yl_train,
            self.yl_test,
            self.yu_train,
            self.yu_test,
        ) = train_test_split(self.x, self.yl, self.yu)


def analyze_and_plot(df, dataset_label):
    data = EmpiricalData(df)

    # Bandwidth selection
    candidate_bandwidth = 0.5 * np.arange(1, 4) * silvermans_rule(data.x)
    h_cv, coverage_results, _ = cv_bandwidth(data, candidate_bandwidth, alpha=0.05)

    # Prediction interval calculation
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
    scores = scores[~np.isnan(scores)]
    qq = np.quantile(scores, [0.95], method="higher")

    # Visualization for Education
    edu_variable = np.arange(8, 20)
    exp_fixed = np.full_like(edu_variable, 10)
    x_eval_fixed = np.column_stack((edu_variable, exp_fixed))
    pred_interval_eval = pred_interval(
        x_eval_fixed,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=h_cv,
    )
    conformal_interval_eval_edu = np.array(
        [pred_interval_eval[0] - qq, pred_interval_eval[1]] + qq
    )

    # Visualization for Experience
    exp_variable = np.arange(df["Exp"].min(), df["Exp"].max())
    edu_fixed = np.full_like(exp_variable, 10)
    x_eval_fixed = np.column_stack((edu_fixed, exp_variable))
    pred_interval_eval = pred_interval(
        x_eval_fixed,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=h_cv,
    )
    conformal_interval_eval_exp = np.array(
        [pred_interval_eval[0] - qq, pred_interval_eval[1]] + qq
    )

    plot_results = {
        "edu": (edu_variable, conformal_interval_eval_edu, exp_fixed[0]),
        "exp": (exp_variable, conformal_interval_eval_exp, edu_fixed[0]),
    }
    return data, plot_results


# Analyze and plot for cleaned_data
df_cleaned = pd.read_csv("wage-data/cleaned_data.csv")
data_cleaned, plot_result_all = analyze_and_plot(df_cleaned, "cleaned_data")

# Analyze and plot for cleaned_data_netwk
df_cleaned_netwk = pd.read_csv("wage-data/cleaned_data_netwk.csv")
data_cleaned_netwk, plot_result_netwk = analyze_and_plot(
    df_cleaned_netwk, "cleaned_data_netwk"
)
# %%


plt.plot(
    plot_result_all["edu"][0],
    plot_result_all["edu"][1][0],
    label=f"All - Conformal Lower bound",
)
plt.plot(
    plot_result_all["edu"][0],
    plot_result_all["edu"][1][1],
    label=f"All - Conformal Upper bound",
)

plt.plot(
    plot_result_netwk["edu"][0],
    plot_result_netwk["edu"][1][0],
    label=f"Netwk - Conformal Lower bound",
)
plt.plot(
    plot_result_netwk["edu"][0],
    plot_result_netwk["edu"][1][1],
    label=f"Netwk - Conformal Lower bound",
)
    

plt.xlabel("Education")
plt.ylabel("Predicted earnings")
plt.legend()
plt.savefig("conformal_intervals_empirical_edu.pdf")
plt.title(f"Conformal Prediction Intervals when Experience is fixed at {plot_result_all['edu'][2]}")

# %% 


plt.plot(
    plot_result_all["exp"][0],
    plot_result_all["exp"][1][0],
    label=f"All - Conformal Lower bound",
)
plt.plot(
    plot_result_all["exp"][0],
    plot_result_all["exp"][1][1],
    label=f"All - Conformal Upper bound",
)

plt.plot(
    plot_result_netwk["exp"][0],
    plot_result_netwk["exp"][1][0],
    label=f"Netwk - Conformal Lower bound",
)
plt.plot(
    plot_result_netwk["exp"][0],
    plot_result_netwk["exp"][1][1],
    label=f"Netwk - Conformal Lower bound",
)

plt.xlabel("Experience")
plt.ylabel("Predicted earnings")
plt.legend()
plt.savefig("conformal_intervals_empirical_exp.pdf")
plt.title(f"Conformal Prediction Intervals when Education is fixed at {plot_result_all['exp'][2]}")
# %%
