# %%
import numpy as np
from utils_dgp import SimData

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso


def cal_y_signal(x, beta, pos):
    return x[:, pos] @ beta


# %%
nfeature = 2000
nsample = 100
sim = SimData(N1=nsample, K=nfeature, scale=3, x_distri="uniform")

sim.x = sim.gen_x(N=nsample, K=nfeature)
sim.epsilon = sim.gen_noise(N=nsample)

pos = [0, 1, 2, 3, 4]
beta = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])[pos]

sim.y_signal = cal_y_signal(sim.x, beta, pos)
sim.y = sim.y_signal + sim.epsilon
sim.yl, sim.yu = sim.get_intervals(sim.y)

# %%
lasso = Lasso(alpha=0.2).fit(sim.x, sim.yl)
lasso_coef = lasso.coef_

print(lasso_coef[pos])

# %%

lasso1 = Lasso(alpha=0.1)
lasso1.fit(sim.x, sim.y)
lasso_coef1 = lasso1.coef_
plt.plot(range(len(lasso_coef1)), lasso_coef1)


# %%

lasso_coef2_array = []
for j in range(sim.M):
    lasso2 = Lasso(alpha=0.1)
    lasso2.fit(sim.x, sim.yd[j])
    lasso_coef2 = lasso2.coef_
    lasso_coef2_array.append(lasso_coef2)

lasso_coef2_array = np.array(lasso_coef2_array)

# %%
mean_coef = np.mean(lasso_coef2_array, axis=0)
plt.plot(
    range(len(mean_coef) - len(pos)),
    mean_coef[~np.isin(range(len(mean_coef)), pos)],
    label="mean",
)
plt.plot(
    range(len(lasso_coef) - len(pos)),
    lasso_coef[~np.isin(range(len(lasso_coef)), pos)],
    label="lasso on yl",
)
plt.ylim(-0.4, 0.4)
plt.legend()


# %%
def compute_lasso_coef(x, y, alpha=0.1):
    lasso = Lasso(alpha=alpha)
    lasso.fit(x, y)
    lasso_coef = lasso.coef_
    return lasso_coef


def plot_lasso_coef(lasso_coef, pos, label="lasso coeff"):
    plt.plot(
        range(len(lasso_coef) - len(pos)),
        lasso_coef[~np.isin(range(len(lasso_coef)), pos)],
        alpha=0.5,
        label=label,
    )
    plt.ylim(auto=True)
    plt.legend()


m = 1
lasso_coefm = compute_lasso_coef(sim.x, sim.yd[m])
# plot_lasso_coef(lasso_coefm, pos, label="lasso on yd{m}")
plot_lasso_coef(lasso_coef, pos, label="lasso on yl")
# plot_lasso_coef(lasso1.coef_, pos, label="lasso on y")
plot_lasso_coef(mean_coef, pos, label="mean of lasso on yd")
# %%
from sklearn.kernel_ridge import KernelRidge

krr = KernelRidge(alpha=0.1, kernel="rbf")
yl_pred = krr.fit(sim.x, sim.yl).predict(sim.x)
# %%
plt.plot(sim.x[:, 1], sim.yl, "o", label="y")
plt.plot(sim.x[:, 1], yl_pred, "o", label="yl_pred")
# %%
