# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wlpy.regression import LocLin


def compute_score(y_test_prediction, yu_test, yl_test):
    diff_l = np.maximum(yl_test - y_test_prediction, 0) ** 2
    diff_u = np.maximum(y_test_prediction - yu_test, 0) ** 2
    return (diff_l + diff_u).mean(axis=1)


def cal_y_signal(eval, params):
    return 2.3 + np.dot(eval, params) + eval[:, -1] ** 2 + np.sin(eval[:, -1])


# %%
n = 20000
M = 1000
k = 1
beta = np.ones(k)*0.3
var_epsilon = 0.0

# x = np.random.normal(0, 1, (n, k))
x = np.random.uniform(0, 1, (n, k))
# epsilon = np.random.normal(0, var_epsilon, n)
df=1
epsilon = (np.random.chisquare(df, n)-df)/np.sqrt(2*df)
# epsilon = 0
y = cal_y_signal(x, beta) + epsilon

# %% 
# Plot the empirical distribution of y conditional on it falls into one of the intervals 
a =2
b =3
y_in_a_b = y[((y>= a) & (y <= b))]
plt.figure()
plt.hist(y_in_a_b, bins=100, density=True)
plt.hist(y, bins=100, density=True)
plt.show()
# %% 
# %%
# now i want to generate yl and yu such that yl <= y <= yu
a1 = 0.0
a2 = 0.0


def get_interval(y, a1, a2):
    yl = np.floor(y) - a1
    yu = np.ceil(y) + a2
    return yl, yu


yl, yu = get_interval(y, a1, a2)
y_middle = (yl+ yu)/2
plt.hist(y_middle, bins=100, density=True)

# yd = np.random.uniform(yl, yu, (M, n))
weights = np.random.uniform(0 , 1, M)
yd = np.outer(weights, yl) + np.outer(1 - weights, yu)
# yd = yl

# %%
(
    x_train,
    x_test,
    yu_train,
    yu_test,
    yl_train,
    yl_test,
    y_train,
    y_test,
    yd_train,
    yd_test,
) = train_test_split(x, yu, yl, y, yd.T,  test_size=0.5, random_state=42)


# %% 
model_linear = LinearRegression().fit(x_train, yd_train)
# TODO: might want to parallel compute this part.
# model_kernel = [KernelReg(j, x_train, var_type=f"{'c'*k}") for j in yd_train]
y_test_pred_linear = model_linear.predict(x_test).T
# %%
# model_local_linear = [LocLin(x_train, j) for j in yd_train]
# y_test_pred_local_linear = np.column_stack(
# [mm.vec_fit(x_test) for mm in model_local_linear]
# ).T
# print(y_test_pred_local_linear.shape)
# %%
y_test_pred = y_test_pred_linear
score = compute_score(y_test_pred, yu_test, yl_test)
tolerance = 1 / np.sqrt(n)
# tolerance = 1 / n
# tolerance = 1e-10
# find the smallest score and all the indices that are less than the smallest score + tolerance
smallest_score = np.min(score)
threshold = smallest_score + tolerance
indices = np.where(score < threshold)[0]
print(f"number of indices slected is {indices.shape[0]}")
y_test_prediction_selected = y_test_pred[indices]
# plt.plot(x[:, -1], y_test_prediction_selected)
# %%
n_test_points = 100
evaluation_points = np.column_stack(
    (np.ones((n_test_points, k - 1)), np.linspace(0, np.max(x[:, -1]), n_test_points))
)

y_signal = cal_y_signal(evaluation_points, beta)

plt.figure()
plt.plot(evaluation_points[:, -1], y_signal, label="y_signal")
yl_signal, yu_signal = get_interval(y_signal, a1, a2)
plt.plot(evaluation_points[:, -1], yl_signal, label="y_interval")
plt.plot(evaluation_points[:, -1], yu_signal, label="y_interval")

y_pred = model_linear.predict(evaluation_points).T
# y_pred = np.column_stack([mm.vec_fit(evaluation_points) for mm in model_local_linear]).T

plt.plot(
    evaluation_points[:, -1],
    y_pred.min(axis=0),
    color="red",
    label="pre-selection prediction",
)
plt.plot(evaluation_points[:, -1], y_pred.max(axis=0), color="red")
y_prediction_selected = y_pred[indices]

plt.plot(
    evaluation_points[:, -1],
    y_prediction_selected.min(axis=0),
    "+",
    color="orange",
    label="post-section prediction",
)
plt.plot(
    evaluation_points[:, -1], y_prediction_selected.max(axis=0), "+", color="orange"
)
plt.xlabel("$x_5$")
plt.legend()
y_pred_middle = LinearRegression().fit(x, y_middle)
plt.plot(evaluation_points[ :,-1], y_pred_middle.predict(evaluation_points), '*')

plt.savefig(f"test1-2023-11-23-{n}-{a1}.pdf")
plt.show()

# %%

plt.plot(evaluation_points[:,-1], y_signal)
# %%
