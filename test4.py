# %%
import numpy as np
from utils_dgp import SimData, compute_score
from wlpy.gist import current_time


def run_simulation(est_kwargs, sample_size=1000):
    sim = SimData(N1=sample_size)
    sim.generate_data()
    sim.calculate_weights_and_yd()
    sim.split_data()
    sim.fit_and_predict(**est_kwargs)
    return sim


est_kwargs = {
    "method": "krr",
    "krr_kernel": "rbf",
    "krr_alpha": 0.05,
    "rf_max_depth": 10,
    "tolerance": 1 / (400) ** 0.7,
}

sim = run_simulation(est_kwargs)

# %% Conformal inference part
sim.score
# %%
sim.y_conformal_pred
conformal_score = compute_score(
    sim.y_conformal_pred, sim.yl_conformal, sim.yu_conformal, option="all"
)

conformal_score = conformal_score[sim.indices].max(axis=0)

# %%
sim.fitted_model.predict(sim.x_eval).T[sim.indices]
# %%
# x_new = np.zeros((1, sim.x_eval.shape[1]))
y_new_range = np.linspace(-5, 10, 200)
qq = np.quantile(conformal_score, 0.95)
print(f"qq: {qq}")
conformal_set = []
for j in range(sim.x_eval.shape[0]):
    x_new = sim.x_eval[[j]]
    y_new_pred = sim.fitted_model.predict(x_new).T[sim.indices]

    y_new_test_score = np.array(
        [
            compute_score(y_new_pred, y_new_test, y_new_test)
            for y_new_test in y_new_range
        ]
    ).max(axis=1)
    selected = np.where(y_new_test_score < qq)[0]
    conformal_set += [[y_new_range[selected].min(), y_new_range[selected].max()]]
conformal_set = np.array(conformal_set).T


# %%

print(conformal_set)
import matplotlib.pyplot as plt

# %%
sim.fitted_model.fit(sim.x, sim.y).predict(sim.x_conformal)
score_if_y_observed = np.abs(
    sim.y_conformal - sim.fitted_model.fit(sim.x, sim.y).predict(sim.x_conformal)
)

qq2 = np.quantile(score_if_y_observed, 0.95)
y_hat = sim.fitted_model.fit(sim.x, sim.y).predict(sim.x_eval)

# %%

res_min, res_max = sim.res_min, sim.res_max
plt.plot(sim.x_eval[:, -1], sim.y_eval_signal, linewidth=2.5, label="true signal")

# plt.plot(sim.x_eval[:, -1], res_min, "-", label="pred error min")
# plt.plot(sim.x_eval[:, -1], res_max, "-", label="pred error max")
# plt.plot(
# sim.x_eval[:, -1],
# np.abs(sim.y_eval_signal - sim.y_true_fit),
# "-",
# label="pred error when we observe y",
# )

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
    f"$N$={sim.N1}, $M$={sim.M}, $K$={sim.K}, $v_\epsilon$={sim.var_epsilon}, $b_1$={sim.interval_bias[0]}, $b_2$={sim.interval_bias[1]}, \n $x$_distri={sim.x_distri}, $\epsilon$_distri={sim.epsilon_distri}, df={sim.df}, scale = {sim.scale}"
)

plt.fill_between(
    sim.x_eval[:, -1],
    conformal_set[0],
    conformal_set[1],
    color="gray",
    alpha=0.5,
    label="conformal set",
)
plt.fill_between(
    sim.x_eval[:, -1],
    y_hat - qq2,
    y_hat + qq2,
    alpha=0.5,
    label="conformal set when observe y",
)

plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.savefig(f"simulation-results/{current_time()}-pred_error.pdf", bbox_inches="tight")

# %%
