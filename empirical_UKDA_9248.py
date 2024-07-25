# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from cross_validation import cv_bandwidth
from sklearn.kernel_ridge import KernelRidge

filename = "apsp_jd23_eul_pwta22.tab"


def analyze_and_plot(
    df,
    x_cols=["Education", "Experience"],
    yl_col="Log_lower_bound",
    yu_col="Log_upper_bound",
    alpha=0.05,
    edu_fixed=14,
    exp_fixed=20,
    edu_variable=np.array([11, 12, 14, 16, 18]),
    exp_variable=np.array([10, 11, 12, 13, 14, 16, 17, 18, 19]),
    conformal_method="local",
    bandwidth=np.array([0.5, 3.0]),
):
    data = EmpiricalData(df, x_cols, yl_col, yu_col)

    if bandwidth is None:
        h_cv = cvBandwidthSelection(alpha, data)
    else:
        h_cv = bandwidth
    # h_cv = np.array([1.1, 2.5])
    print(f"The bandwidth is {h_cv}")

    exp_arr = np.full_like(edu_variable, exp_fixed)
    x_eval_fixed = np.column_stack((edu_variable, exp_arr))

    if conformal_method == "split":
        # Prediction interval calculation
        pred_interval_test = pred_interval(
            data.x_test.to_numpy(),
            data.x_train,
            data.yl_train,
            data.yu_train,
            h=h_cv,
            alpha=alpha,
        )
        scores = np.maximum(
            pred_interval_test[0] - data.yl_test, data.yu_test - pred_interval_test[1]
        )
        scores = scores[~np.isnan(scores)]
        print(scores)
        qq = np.quantile(scores, [1 - alpha], method="higher")

    if conformal_method == "local":
        qq = np.zeros(edu_variable.shape)
        for j, edu in enumerate(edu_variable):
            condition = (
                (data.df[x_cols[0]] == edu)
                & (data.df[x_cols[1]] >= (exp_fixed - 2))
                & (data.df[x_cols[1]] < (exp_fixed + 2))
                & (data.df["is_test"] == 1)
            )
            print(j, edu, sum(condition))

            x_test_locl = data.df.loc[condition, x_cols].to_numpy()
            yl_test_local = data.df.loc[condition, yl_col].to_numpy()
            yu_test_locl = data.df.loc[condition, yu_col].to_numpy()

            pred_interval_test = pred_interval(
                x_test_locl,
                data.x_train,
                data.yl_train,
                data.yu_train,
                h=h_cv,
                alpha=alpha,
            )
            scores = np.maximum(
                pred_interval_test[0] - yl_test_local,
                yu_test_locl - pred_interval_test[1],
            )
            plt.figure()
            plt.hist(scores, bins=100)
            plt.show()
            qvalue = np.quantile(scores, 1 - alpha)
            qq[j] = qvalue
            # print(qq, qq[j], np.quantile(scores, [0.5, 0.25, 0.75, 1 - alpha]))
    print(qq)
    pred_interval_eval_edu = pred_interval(
        x_eval_fixed,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=h_cv,
    )
    conformal_interval_eval_edu = np.array(
        [pred_interval_eval_edu[0] - qq, pred_interval_eval_edu[1] + qq]
    )

    # Visualization for Experience
    # exp_variable = np.arange(df["Experience"].min(), df["Experience"].max(), step=3)

    # edu_arr = np.full_like(exp_variable, edu_fixed)
    # x_eval_fixed = np.column_stack((edu_arr, exp_variable))
    # pred_interval_eval_exp = pred_interval(
    #     x_eval_fixed,
    #     data.x_train,
    #     data.yl_train,
    #     data.yu_train,
    #     h=h_cv,
    # )
    # conformal_interval_eval_exp = np.array(
    #     [pred_interval_eval_exp[0] - qq, pred_interval_eval_exp[1] + qq]
    # )

    result_conformal = {
        "edu": (edu_variable, conformal_interval_eval_edu, exp_arr[0]),
        # "exp": (exp_variable, conformal_interval_eval_exp, edu_arr[0]),
    }
    result_pred = {
        "edu": (edu_variable, pred_interval_eval_edu, exp_arr[0]),
        # "exp": (exp_variable, pred_interval_eval_exp, edu_arr[0]),
    }
    return data, result_pred, result_conformal


def cvBandwidthSelection(alpha, data):
    h_silverman = np.array(
        [silvermans_rule(data.x[:, [0]]), silvermans_rule(data.x[:, [1]])]
    )
    # Bandwidth selection
    candidate_bandwidth = np.array(
        [h_silverman * 0.15 * j for j in np.linspace(1, 10, num=10)]
    )
    #
    h_cv, coverage_results, _ = cv_bandwidth(
        data, candidate_bandwidth, alpha=alpha, nfolds=5
    )

    return h_cv


def visualize_prediction(
    result, transform, label, color, variable="edu", offset=0, marker="o"
):
    x = result[variable][0]
    yl = transform(result[variable][1][0])
    yu = transform(result[variable][1][1])

    # Plot the line segments with labels
    plt.plot(
        [], [], color=color, marker=marker, linestyle="-", linewidth=1, label=label
    )  # Empty plot for the legend entry

    for xi, yli, yui in zip(x + offset, yl, yu):
        plt.plot([xi, xi], [yli, yui], color=color, linestyle="-", linewidth=1)
        plt.plot(
            xi, yli, marker=marker, markersize=4, color=color
        )  # Symbol at lower endpoint
        plt.plot(
            xi, yui, marker=marker, markersize=4, color=color
        )  # Symbol at upper endpoint


# Analyze and plot for cleaned_data
df = pd.read_csv(f"wage-data/clean_{filename}")

df["Log_upper_bound"].describe()
# Analyze and plot for cleaned_data_netwk
# df_cleaned_netwk = pd.read_csv("wage-data/cleaned_data_netwk.csv")
# dt_netwk, result_netwk = analyze_and_plot(
#     df_cleaned_netwk, "cleaned_data_netwk"
# )
# %%

alpha = 0.5
dt, result_all, result_all_conformal = analyze_and_plot(
    df,
    alpha=alpha,
)

dt_exact, result_exact, result_exact_conformal = analyze_and_plot(
    df.loc[~df["range_indicator"]], alpha=alpha, conformal_method="local"
)

# dt_imputed, result_imputed, result_imputed_conformal = analyze_and_plot(
#     df,
#     yl_col="Log_income_with_imputed_lower",
#     yu_col="Log_income_with_imputed_upper",
#     alpha=alpha,
# )
# %%

transform = np.array


plt.figure()

visualize_prediction(
    result_all,
    transform,
    f"Prediction with exact number and range data",
    "tab:blue",
    offset=-0.4
)
visualize_prediction(
    result_all_conformal,
    transform,
    f"Prediction with exact number and range data (conformalised)",
    "tab:red",
    offset=-0.2,
)
print(
    np.exp(result_all_conformal["edu"][1][0]),
    np.exp(result_all_conformal["edu"][1][1]),
)

visualize_prediction(
    result_exact,
    transform,
    f"Prediction with exact number data",
    "tab:orange",
)
visualize_prediction(
    result_exact_conformal,
    transform,
    f"Prediction with exact number data (conformalised)",
    "tab:green",
    offset=0.2,
    marker="s",
)
print(
    np.exp(result_exact_conformal["edu"][1][0]),
    np.exp(result_exact_conformal["edu"][1][1]),
)
plt.xlabel("Education")
plt.ylabel("Predicted log earnings")
# plt.legend(loc = "right")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {result_all['edu'][2]}"
)
plt.savefig("conformal_intervals_empirical_edu.pdf")


# %%
