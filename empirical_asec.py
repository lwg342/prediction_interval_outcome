# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from cross_validation import cv_bandwidth
from sklearn.kernel_ridge import KernelRidge


def analyze_and_plot(
    df,
    x_cols=["Edu", "Experience"],
    yl_col="log_lower_bound",
    yu_col="log_upper_bound",
    alpha=0.05,
    edu_fixed=14,
    exp_fixed=15,
    edu_variable=np.array([10, 12, 14, 16, 18]),
    exp_variable=np.array([10, 11, 12, 13, 14, 16, 17, 18, 19]),
    conformal_method="local",
):
    data = EmpiricalData(df, x_cols, yl_col, yu_col)

    # Bandwidth selection
    candidate_bandwidth = 0.3 * np.arange(1, 5) * silvermans_rule(data.x)
    # h_cv, coverage_results, _ = cv_bandwidth(
    #     data, candidate_bandwidth, alpha=0.1, nfolds=5
    # )

    # h_cv = np.array(
    # [silvermans_rule(data.x[:, [0]]) + 0.1, silvermans_rule(data.x[:, [1]])]
    # )
    h_cv = np.array([0.1, 1.6])
    print(f"The bandwidth is {h_cv}")

    exp_arr = np.full_like(edu_variable, exp_fixed)
    x_eval_fixed = np.column_stack((edu_variable, exp_arr))

    if conformal_method == "split":
        # Prediction interval calculation
        pred_interval_test = pred_interval(
            data.x_test, data.x_train, data.yl_train, data.yu_train, h=h_cv, alpha=alpha
        )
        scores = np.maximum(
            pred_interval_test[0] - data.yl_test, data.yu_test - pred_interval_test[1]
        )
        scores = scores[~np.isnan(scores)]
        print(scores)
        qq = np.quantile(scores, [1 - alpha], method="higher")

    if conformal_method == "local":
        qq = np.zeros_like(edu_variable)
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
            qq[j] = np.quantile(scores, [1 - alpha], method="higher")
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

def visualize_prediction(result, transform, label, color, variable="edu", offset=0, marker="o"):
    x = result[variable][0]
    yl = transform(result[variable][1][0])
    yu = transform(result[variable][1][1])
    
    # Plot the line segments with labels
    plt.plot([], [], color=color, marker=marker, linestyle='-', linewidth=1, label=label)  # Empty plot for the legend entry
    
    for xi, yli, yui in zip(x + offset, yl, yu):
        plt.plot([xi, xi], [yli, yui], color=color, linestyle="-", linewidth=1)
        plt.plot(
            xi, yli, marker=marker, markersize=4, color=color
        )  # Symbol at lower endpoint
        plt.plot(
            xi, yui, marker=marker, markersize=4, color=color
        )  # Symbol at upper endpoint


# Analyze and plot for cleaned_data
df = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")

df["log_upper_bound"].describe()
# Analyze and plot for cleaned_data_netwk
# df_cleaned_netwk = pd.read_csv("wage-data/cleaned_data_netwk.csv")
# dt_netwk, result_netwk = analyze_and_plot(
#     df_cleaned_netwk, "cleaned_data_netwk"
# )
# %%

alpha = 0.1
dt, result_all, result_all_conformal = analyze_and_plot(
    df,
    alpha=alpha,
)

dt_exact, result_exact, result_exact_conformal = analyze_and_plot(
    df.loc[df["I_ERNVAL"] == 0], alpha=alpha, conformal_method="local"
)

dt_imputed, result_imputed, result_imputed_conformal = analyze_and_plot(
    df,
    yl_col="log_income_with_imputed_lower",
    yu_col="log_income_with_imputed_upper",
    alpha=alpha,
)
# %%

transform = np.array


plt.figure()

# visualize_prediction(
#     result_all,
#     transform,
#     f"Prediction with exact number and range data",
#     "tab:blue",
# )
visualize_prediction(
    result_all_conformal,
    transform,
    f"Prediction with exact number and range data",
    "tab:red",
    offset=-0.2,
)
print(
    np.exp(result_all_conformal["edu"][1][0]),
    np.exp(result_all_conformal["edu"][1][1]),
)

# visualize_prediction(
#     result_exact,
#     transform,
#     f"Prediction with exact number data",
#     "tab:orange",
# )
visualize_prediction(
    result_exact_conformal,
    transform,
    f"Prediction with exact number data",
    "tab:green",
    offset=0, 
    marker="s"
)
# visualize_prediction(
#     result_imputed,
#     transform,
#     f"Prediction with exact number data and imputed data",
#     "tab:orange",
# )
visualize_prediction(
    result_imputed_conformal,
    transform,
    f"Prediction with exact number data and imputed data",
    "tab:purple",
    offset=0.2,
    marker="x"
)


plt.xlabel("Education")
plt.ylabel("Predicted log earnings")
plt.legend()
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {result_all['edu'][2]}"
)
plt.savefig("conformal_intervals_empirical_edu.pdf")


# %%
Iu = df["upper_bound"].to_numpy()
plt.hist(Iu, bins=500)
# %%
plt.hist(Iu, bins=500)
plt.xlim(0, 500000)
# %%
sum(Iu == 400000)
# %%
# Local conditions


def chack_locality(arr, val1, val2):
    return (arr[:, 0] == val1) & (val2[0] <= arr[:, 1]) & (arr[:, 1] < val2[1])


start = 0
end = 50
num_intervals = 10  # Adjust this to the desired number of intervals
points = np.linspace(start, end, num_intervals + 1)
exp_intvl = [(points[i], points[i + 1]) for i in range(len(points) - 1)]


dt = EmpiricalData(df, ["Edu", "Experience"], "log_lower_bound", "log_upper_bound")

local_condition = [
    lambda x: chack_locality(x, i, j)
    for i in np.array([10, 11, 12, 14, 16, 17, 18])
    for j in [(10, 15)]
]
for j in range(10):
    if local_condition[j](dt.x_test)[0]:
        print(j)

sum(local_condition[0](dt.x_test))
dt.x_test[((local_condition[0](dt.x_test)))]

dt.x_test[local_condition[2](dt.x_test)]
# %%
