import numpy as np
from sklearn.model_selection import train_test_split
from utils.cross_validation import cv_bandwidth, cvBandwidthSelection
from . import pred_interval
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge


class EmpiricalData:
    def __init__(
        self,
        df,
        x_cols=["Education", "Experience"],
        yl_col="Log_lower_bound",
        yu_col="Log_upper_bound",
    ):
        self.x = df[x_cols].to_numpy()
        self.yl = df[yl_col].to_numpy()
        self.yu = df[yu_col].to_numpy()
        self.df_train, self.df_test = train_test_split(df)
        df.loc[:, "is_test"] = df.index.isin(self.df_test.index).copy()
        self.df = df

        self.x_train = self.df_train[x_cols].to_numpy()
        self.yl_train = self.df_train[yl_col].to_numpy()
        self.yu_train = self.df_train[yu_col].to_numpy()

        self.x_test = self.df_test[x_cols].to_numpy()
        self.yl_test = self.df_test[yl_col].to_numpy()
        self.yu_test = self.df_test[yu_col].to_numpy()


def analyze_and_plot(
    df,
    x_cols=["Education", "Experience"],
    yl_col="Log_lower_bound",
    yu_col="Log_upper_bound",
    alpha=0.1,
    exp_fixed=15,
    edu_variable=np.array([12, 14, 16, 18]),
    conformal_method="local",
    bandwidth=np.array([0.1, 1.7]),
    comment="",
):
    data = EmpiricalData(df, x_cols, yl_col, yu_col)

    if bandwidth is None:
        h_cv = cvBandwidthSelection(alpha, data)
    else:
        h_cv = bandwidth
    # print(f"The bandwidth is {h_cv}, data comment: {comment}")

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
        qq = np.quantile(scores, [1 - alpha], method="higher")

    if conformal_method == "local":
        qq = np.zeros(edu_variable.shape)
        for j, edu in enumerate(edu_variable):
            condition = (
                (data.df[x_cols[0]] == edu)
                & (data.df[x_cols[1]] >= (exp_fixed - 5))
                & (data.df[x_cols[1]] < (exp_fixed + 5))
                & (data.df["is_test"] == 1)
            )
            # print(j, edu, sum(condition))

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
            # plt.figure()
            # plt.hist(scores, bins=100)
            # plt.show()
            qvalue = np.quantile(scores, 1 - alpha)
            qq[j] = qvalue
    # print(qq)
    if conformal_method == "none":
        qq = 0

    # print("x_eval_fixed", x_eval_fixed)
    pred_interval_eval_edu = pred_interval(
        x_eval_fixed,
        data.x_train,
        data.yl_train,
        data.yu_train,
        h=h_cv,
        alpha=alpha,
    )
    conformal_interval_eval_edu = np.array(
        [pred_interval_eval_edu[0] - qq, pred_interval_eval_edu[1] + qq]
    )

    reg_model_l = KernelRidge(kernel="rbf").fit(data.x, data.yl)
    yl_eval = reg_model_l.predict(x_eval_fixed)
    reg_model_u = KernelRidge(kernel="rbf").fit(data.x, data.yu)
    yu_eval = reg_model_u.predict(x_eval_fixed)

    results = {
        "Education": edu_variable,
        "Experience": [exp_arr[0]] * len(edu_variable),
        "Alpha": [alpha] * len(edu_variable),
        "Prediction Lower Bound": pred_interval_eval_edu[0],
        "Prediction Upper Bound": pred_interval_eval_edu[1],
        "Conformal Prediction Lower Bound": conformal_interval_eval_edu[0],
        "Conformal Prediction Upper Bound": conformal_interval_eval_edu[1],
        "Conformal Correction": qq,
        "Kernel Regression Lower": yl_eval,
        "Kernel Regression Upper": yu_eval,
        "Comment": [comment] * len(edu_variable),
    }
    # print(results)
    return data, results


def visualize_prediction(
    result, key, transform, label, color, offset=0, marker="o", markersize=5
):
    x = result["edu"]
    yl = transform(result[key][0])
    yu = transform(result[key][1])

    # Plot the line segments with labels
    plt.plot(
        [], [], color=color, marker=marker, linestyle="-", linewidth=1, label=label
    )  # Empty plot for the legend entry

    for xi, yli, yui in zip(x + offset, yl, yu):
        plt.plot([xi, xi], [yli, yui], color=color, linestyle="-", linewidth=1)
        plt.plot(
            xi, yli, marker=marker, markersize=markersize, color=color
        )  # Symbol at lower endpoint
        plt.plot(
            xi, yui, marker=marker, markersize=markersize, color=color
        )  # Symbol at upper endpoint
