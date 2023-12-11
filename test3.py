# %%
import numpy as np
from utils_dgp import SimData
from wlpy.gist import current_time

# %%
sample_size = 400
sim = SimData(N=sample_size, epsilon_distri="chisquare")
sim.generate_data()
sim.calculate_weights_and_yd()
sim.split_data()

est_kwargs = {
    "method": "krr",
    "krr_kernel": "rbf",
    "krr_alpha": 0.05,
    "rf_max_depth": 10,
    # "tolerance": 0,
    # "tolerance": 1 / np.sqrt(sample_size),
    "tolerance": 1 / (sample_size) ** 0.7,
}

sim.fit_and_predict(**est_kwargs)

sim.indices


# %%
def pred_error(y_true, y_pred):
    res_min = np.min(np.abs(y_true - y_pred), axis=0)
    res_max = np.max(np.abs(y_true - y_pred), axis=0)
    return res_min, res_max


# %%
import matplotlib.pyplot as plt

res_min, res_max = pred_error(
    sim.y_eval_signal,
    sim.y_eval_pred[sim.indices],
)

plt.plot(sim.x_eval[:, -1], sim.y_eval_signal, linewidth=2.5, label="true signal")

plt.plot(sim.x_eval[:, -1], res_min, "-", label="pred error min")
plt.plot(sim.x_eval[:, -1], res_max, "-", label="pred error max")
plt.plot(
    sim.x_eval[:, -1],
    np.abs(sim.y_eval_signal - sim.y_true_fit),
    "-",
    label="pred error when we observe y",
)

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
    sim.y_true_fit,
    linestyle="dashed",
    label="prediction when we observe y",
)
plt.title(
    f"$N$={sim.N}, $M$={sim.M}, $K$={sim.K}, $v_\epsilon$={sim.var_epsilon}, $b_1$={sim.interval_bias[0]}, $b_2$={sim.interval_bias[1]}, \n $x$_distri={sim.x_distri}, $\epsilon$_distri={sim.epsilon_distri}, df={sim.df}, scale = {sim.scale}"
)
plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.savefig(f"simulation-results/{current_time()}-pred_error.pdf", bbox_inches="tight")
# %%
plt.plot(sim.score, ".", label="Loss for each draw")

plt.xlabel(f"Draw m, total number of draws is M = {sim.M}")
plt.title(
    f"$N$={sim.N}, $M$={sim.M}, $K$={sim.K}, $v_\epsilon$={sim.var_epsilon}, $b_1$={sim.interval_bias[0]}, $b_2$={sim.interval_bias[1]}, \n $x$_distri={sim.x_distri}, $\epsilon$_distri={sim.epsilon_distri}, df={sim.df}, scale = {sim.scale}"
)
plt.axhline(y=sim.score.min() + 1 / sim.N**0.5, color = "tab:green", linestyle="--", label="tolerance = n^-0.5")
plt.axhline(
    y=sim.score.min() + est_kwargs["tolerance"],
    color="tab:red",
    linestyle="--",
    label="tolerance = n^-0.7",
)
plt.axhline(y=sim.score.min() + 1 / sim.N, color = "tab:orange", linestyle="--", label="tolerance = n^-1")
plt.legend()
plt.savefig(f"simulation-results/{current_time()}-loss.pdf", bbox_inches="tight")
# %%
