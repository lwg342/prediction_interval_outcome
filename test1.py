# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def compute_score(y_test_prediction, yu_test, yl_test):
    diff_l = np.maximum(yl_test - y_test_prediction, 0) ** 2
    diff_u = np.maximum(y_test_prediction - yu_test, 0) ** 2
    return (diff_l + diff_u).mean(axis=1)


# %%
n = 20000
M = 500
k = 5
beta = np.ones(k)
var_epsilon = 1.0

x = np.random.normal(0, 1, (n, k))
# x = np.random.uniform(0, 3.4, (n, k))
epsilon = np.random.normal(0, var_epsilon, n)
y = 2.0 + np.dot(x, beta) + epsilon

# %%
# now i want to generate yl and yu such that yl <= y <= yu
a1 = 1.0
a2 = a1
yl = np.floor(y) - a1
yu = np.ceil(y) + a2
yd = np.random.uniform(yl, yu, (M, n))

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
) = train_test_split(x, yu, yl, y, test_size=0.5, random_state=42)
yd_train = np.random.uniform(yl_train, yu_train, (M, len(yu_train)))
model_train = LinearRegression().fit(x_train, yd_train.T)
# %%
y_test_prediction = model_train.predict(x_test).T

# %%
score = compute_score(y_test_prediction, yu_test, yl_test)
tolerance = 1 / (n)
# find the smallest score and all the indices that are less than the smallest score + tolerance
smallest_score = np.min(score)
threshold = smallest_score + tolerance
indices = np.where(score < threshold)[0]
print(f"number of indices slected is {indices.shape[0]}")
y_test_prediction_selected = y_test_prediction[indices]

# %%
n_test_points = 100
evaluation_points = np.column_stack(
    (np.ones((n_test_points, k - 1)), np.linspace(1, 1.5, n_test_points))
)
y_signal = 2.0 + np.dot(evaluation_points, beta)

plt.figure()
plt.plot(evaluation_points[:, -1], y_signal, label="y_signal")

y_pred = model_train.predict(evaluation_points).T
plt.plot(
    evaluation_points[:, -1],
    y_pred.min(axis=0),
    color="red",
    label="pre-selection prediction",
)
plt.plot(evaluation_points[:, -1], y_pred.max(axis=0), color="red")
y_prediction_selected = model_train.predict(evaluation_points).T[indices]
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
plt.savefig(f"test1-2023-11-23-{n}-{a1}.pdf")
plt.show()

# %%
