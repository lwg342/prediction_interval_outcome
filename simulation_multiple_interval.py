# %%
import numpy as np


# Define f(x) and g(x) as per the provided formulas
def f(x):
    return (x - 1) ** 2 * (x + 1)


def g(x):
    return 2 * np.sqrt((x + 0.5) * (x >= -0.5))


# Define sigma^2(x) (variance function), here assumed constant, adjust if needed
def sigma2(x):
    return 0.5  # You can replace this with any other function if required


# Generate X ~ Unif[-1.5, 1.5]
def generate_X(n):
    return np.random.uniform(-1.5, 1.5, size=[n, 1])


# Generate Y
def generate_Y(X):
    Y = []
    for x in X:
        if np.random.rand() < 0.5:
            Y.append(np.random.normal(f(x) - g(x), np.sqrt(sigma2(x))))
        else:
            Y.append(np.random.normal(f(x) + g(x), np.sqrt(sigma2(x))))
    return np.array(Y).flatten()


def get_interval(y, scale, **kwargs):
    err1 = np.abs(np.random.normal(0, scale, y.shape))
    err2 = np.abs(np.random.normal(0, scale, y.shape))
    # err1 =0
    # err2 = 0.2
    return (y - err1, y + err2)


# %%
# [-] Example usage

from utils.simulation import IntervalCensoredData
from utils import *
from utils.set_estimation import *
import matplotlib.pyplot as plt

N = 1000
X = generate_X(N)
Y = generate_Y(X)
YL, YU = get_interval(Y, 0.5)
data = IntervalCensoredData(X, Y, YL, YU)
x, yl, yu = data.x, data.yl, data.yu

x_eval = np.linspace(-1.5, 1.5, 50)[:, np.newaxis]
weights_eval = normalize_weights(
    multivariate_epanechnikov_kernel(x_eval, data.x, h=0.1)
)
weights = weights_eval[0]
# %%
# [-] Example: estimating 1 and multiple pred. intervals


grid = create_grid(*eligible_t0t1(yl, yu, weights), n_grid=100)
K = 2
interval_arr = create_valid_interval_array(grid, grid)

optim_interval = find_optimal_interval(interval_arr, weights, yl, yu, alpha=0.1)
print(optim_interval)
print(find_optimal_set(interval_arr, weights, yl, yu, alpha=0.1))

# %%
# [-] Example: estimating pred. set over x_eval
K = 2
rslt = np.zeros((len(x_eval), 2 * K))
weights_eval = normalize_weights(multivariate_epanechnikov_kernel(x_eval, data.x, 0.2))
for i, x_e in enumerate(x_eval):
    weights = weights_eval[i]
    grid = create_grid(*eligible_t0t1(yl, yu, weights), n_grid=100)
    interval_arr = create_valid_interval_array(grid, grid)
    if K == 1:
        optim_interval = find_optimal_interval(interval_arr, weights, yl, yu, alpha=0.1)
        rslt[i] = optim_interval
    if K == 2:
        optim_set = find_optimal_set(interval_arr, weights, yl, yu, alpha=0.1, K=2)
        rslt[i] = optim_set


# %%
# [-] Plot the intervals in rslt against x_eval


plt.figure(figsize=(10, 6))
for j in range(rslt.shape[1]):
    if j == 0:
        plt.plot(
            x_eval.flatten(), rslt[:, j], color="tab:blue", label="Prediction interval"
        )
    else:
        plt.plot(x_eval.flatten(), rslt[:, j], color="tab:blue")

plt.plot(data.x, data.yl, ".", color="tab:orange", markersize=4)
plt.plot(data.x, data.yu, ".", color="tab:orange", markersize=4)
plt.vlines(
    data.x,
    data.yl,
    data.yu,
    colors="tab:green",
    linestyles="solid",
    alpha=0.3,
    label="Interval (yl,yu)",
)


plt.xlabel("x")
plt.ylabel("y")
plt.title("Intervals against x_eval")
plt.legend()

plt.tight_layout()
plt.show()

# %%
