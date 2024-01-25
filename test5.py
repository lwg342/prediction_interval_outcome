# %%
import numpy as np
from utils_dgp import SimData
import matplotlib.pyplot as plt

sample_size = 10
sim = SimData(N1=sample_size, scale=1, epsilon_distri="normal")
sim.generate_data()
sim.calculate_weights_and_yd()
sim.split_data()

# %%
plt.scatter(sim.x[:, -1], sim.y, s=1, alpha=0.5)
plt.scatter(sim.x[:, -1], sim.yd[0], s=1, alpha=0.5)
plt.scatter(sim.x[:, -1], sim.yl, s=5, alpha=0.5)
plt.scatter(sim.x[:, -1], sim.yu, s=5, alpha=0.5)


# %%

def lin_model(x, params):
    return 10 * np.ones(x.shape[0]) + 1 * x[:, -1]

# %%
sample_size = 2000
sim = SimData(
    N1=sample_size, scale=1, M=5000, epsilon_distri="normal", std_eps=1.0, cal_y_signal=lin_model
)
sim.generate_data()
# weights = np.outer(np.linspace(0, 1, sim.M), np.ones(sim.N1))
weights =None
sim.calculate_weights_and_yd(weights=weights)
sim.split_data()

est_kwargs = {
    "method": "krr",
    "krr_kernel": "rbf",
    "krr_alpha": 0.02,
    "rf_max_depth": 10,
    "rf_n_estimators": 200,
    # "tolerance": 0,
    "tolerance": 1 / np.sqrt(sample_size),
}

sim.fit(**est_kwargs)

sim.indices
# %%
plt.scatter(sim.x[:, -1], sim.y, s=1, alpha=0.5)
plt.scatter(sim.x[:, -1], sim.yl, s=1, alpha=0.5)
plt.scatter(sim.x[:, -1], sim.yu, s=1, alpha=0.5)

plt.fill_between(
    sim.x_eval[:, -1],
    sim.y_eval_pred.min(axis=0),
    sim.y_eval_pred.max(axis=0),
    color="gray",
    alpha=0.5,
    label="pre-selection prediction set",
)
plt.fill_between(
    sim.x_eval[:, -1],
    sim.y_eval_pred[sim.indices].min(axis=0),
    sim.y_eval_pred[sim.indices].max(axis=0),
    color="red",
    alpha=0.5,
    label="post-selection prediction set",
)

plt.plot(
    sim.x_eval[:, -1],
    sim.y_eval_pred_obs,
    linestyle="dashed",
    label="prediction when we observe y",
)
plt.plot(sim.x_eval[:, -1], sim.y_eval_signal, linewidth=2.5, label="true signal")
plt.legend(loc="best")

# %%
plt.plot(sim.score, ".", label="Loss for each draw")

plt.xlabel(f"Draw m, total number of draws is M = {sim.M}")
plt.title(
    f"$N$={sim.N1}, $M$={sim.M}, $K$={sim.K}, $v_\epsilon$={sim.std_eps}, $b_1$={sim.interval_bias[0]}, $b_2$={sim.interval_bias[1]}, \n $x$_distri={sim.x_distri}, $\epsilon$_distri={sim.epsilon_distri}, df={sim.df}, scale = {sim.scale}"
)
plt.axhline(
    y=sim.score.min() + 1 / sim.N1**0.5,
    color="tab:green",
    linestyle="--",
    label="tolerance = n^-0.5",
)
plt.axhline(
    y=sim.score.min() + est_kwargs["tolerance"],
    color="tab:red",
    linestyle="--",
    label="tolerance = n^-0.7",
)
plt.axhline(
    y=sim.score.min() + 1 / sim.N1,
    color="tab:orange",
    linestyle="--",
    label="tolerance = n^-1",
)
plt.legend(loc="best")
# %%
