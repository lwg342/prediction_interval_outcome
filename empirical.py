# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_sim import *
from cross_validation import cv_bandwidth
from sklearn.kernel_ridge import KernelRidge

# read file from wage-data/cleaned_data.csv
df = pd.read_csv("wage-data/cleaned_data_netwk.csv")
# df = pd.read_csv("wage-data/cleaned_data.csv")
df


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


data = EmpiricalData(df)
# %%

candidate_bandwidth = 0.3 * np.arange(1, 10) * silvermans_rule(data.x)

h_cv, coverage_results, _ = cv_bandwidth(data, candidate_bandwidth, alpha=0.05)
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
# keep scores that are not nan
scores = scores[~np.isnan(scores)]
qq = np.quantile(scores, [0.95], method="higher")
print(qq)
# %%
# find the value Exp that has the largest number of observations
df["Edu"].value_counts().idxmax()
# construct x_eval_fixed by setting Edu to range from min of Edu to max of Edu and Exp to 10

edu_fixed = np.arange(df["Edu"].min(), df["Edu"].max())
exp_fixed = np.full_like(edu_fixed, 10)

x_eval_fixed = np.column_stack((edu_fixed, exp_fixed))
pred_interval_eval = pred_interval(
    x_eval_fixed,
    data.x_train,
    data.yl_train,
    data.yu_train,
    h=h_cv,
)
conformal_interval_eval = np.array(
    [pred_interval_eval[0] - qq, pred_interval_eval[1]] + qq
)

plt.plot(edu_fixed, conformal_interval_eval[0], label="Conformal Lower bound")
plt.plot(edu_fixed, conformal_interval_eval[1], label="Conformal Upper bound")
plt.xlabel("Education")
plt.ylabel("Interval")
plt.legend()
plt.title("Conformal Prediction Intervals")
plt.savefig("conformal_intervals_empirical_edu.pdf")

# %%
exp_fixed = np.arange(df["Exp"].min(), df["Exp"].max())
edu_fixed = np.full_like(exp_fixed, 12)

x_eval_fixed = np.column_stack((edu_fixed, exp_fixed))
pred_interval_eval = pred_interval(
    x_eval_fixed,
    data.x_train,
    data.yl_train,
    data.yu_train,
    h=h_cv,
)
conformal_interval_eval = np.array(
    [pred_interval_eval[0] - qq, pred_interval_eval[1]] + qq
)

plt.plot(exp_fixed, conformal_interval_eval[0], label="Conformal Lower bound")
plt.plot(exp_fixed, conformal_interval_eval[1], label="Conformal Upper bound")
plt.xlabel("Experience")
plt.ylabel("Interval")
plt.legend()
plt.title("Conformal Prediction Intervals")
plt.savefig("conformal_intervals_empirical_exp.pdf")
# %%
# Fit kernel regression model
model = KernelRidge(kernel='rbf')
model.fit(data.x_train, data.yl_train)

# Predict on test data
yl_pred = model.predict(x_eval_fixed)
plt.scatter(exp_fixed, yl_pred, label="Predicted Lower Bound")
plt.xlim([0, 50])
# %%
