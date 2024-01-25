# %%
import numpy as np
from utils_dgp import SimData, compute_score
from wlpy.gist import current_time
import numpy as np
import matplotlib.pyplot as plt


# %%
est_kwargs = {
    "method": "krr",
    "krr_kernel": "rbf",
    "krr_alpha": 0.05,
    "rf_max_depth": 10,
    "tolerance": 1 / (400) ** 0.7,
}

sim = SimData(N1=2000, N2=2000, scale=1)
sim.generate_data()
sim.calculate_weights_and_yd()
sim.split_data()
sim.fit_and_predict(**est_kwargs)

# %% Conformal inference part

conformal_score = compute_score(
    sim.y_conformal_pred, sim.yl_conformal, sim.yu_conformal, option="all"
)

conformal_score = conformal_score[sim.indices].min(axis=0)


y_new_range = np.linspace(-5, 15, 1000)
qq = np.quantile(conformal_score, 0.95)
print(f"qq: {qq}")
conformal_set = []
y_new_pred = sim.fitted_model.predict(sim.x_eval).T[sim.indices]
for j in range(sim.x_eval.shape[0]):
    y_new_test_score = np.array(
        [
            compute_score(y_new_pred[:, [j]], y_new_test, y_new_test)
            for y_new_test in y_new_range
        ]
    ).min(axis=1)
    selected = np.where(y_new_test_score < qq)[0]
    conformal_set += [[y_new_range[selected].min(), y_new_range[selected].max()]]
conformal_set = np.array(conformal_set).T
# %% Conformal inference with abs score
score_max = np.abs(sim.y_conformal - sim.y_conformal_pred[sim.indices].max(axis=0))
qq_max = np.quantile(score_max, 0.95)
y_hat_max = sim.y_eval_pred[sim.indices].max(axis=0)
conformal_set_max = y_hat_max + qq_max

score_min = np.abs(sim.y_conformal - sim.y_conformal_pred[sim.indices].min(axis=0))
qq_min = np.quantile(score_min, 0.95)
y_hat_min = sim.y_eval_pred[sim.indices].min(axis=0)
conformal_set_min = y_hat_min - qq_min


# %%
score_if_y_observed = np.abs(sim.y_conformal - sim.y_conformal_pred_obs)

qq2 = np.quantile(score_if_y_observed, 0.95)
y_hat = sim.y_eval_pred_obs

# %%

res_min, res_max = sim.res_min, sim.res_max
plt.plot(sim.x_eval[:, -1], sim.y_eval_signal, linewidth=2.5, label="true signal")

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
plt.title(
    f"$N$={sim.N1}, $M$={sim.M}, $K$={sim.K}, $v_\epsilon$={sim.std_eps}, $b_1$={sim.interval_bias[0]}, $b_2$={sim.interval_bias[1]}, \n $x$_distri={sim.x_distri}, $\epsilon$_distri={sim.epsilon_distri}, df={sim.df}, scale = {sim.scale}"
)

# plt.fill_between(
#     sim.x_eval[:, -1],
#     conformal_set_min,
#     conformal_set_max,
#     # color="gray",
#     alpha=0.5,
#     label="conformal set",
# )
plt.fill_between(
    sim.x_eval[:, -1],
    conformal_set[0],
    conformal_set[1],
    # color="gray",
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
plt.fill_between(
    sim.x_eval[:, -1],
    sim.y_eval_signal - 1.96,
    sim.y_eval_signal + 1.96,
    alpha=0.5,
    label="conformal set y_signal -+ 1.96",
)

plt.legend(loc="best", bbox_to_anchor=(1, 1))
# plt.savefig(f"simulation-results/{current_time()}-pred_error.pdf", bbox_inches="tight")
# %%
y_conformal_pred_max = sim.y_conformal_pred[sim.indices].max(axis=0)
y_conformal_pred_min = sim.y_conformal_pred[sim.indices].min(axis=0)

score_max = compute_score(
    y_conformal_pred_max, sim.yl_conformal, sim.yu_conformal, option="all"
)
score_min = compute_score(
    y_conformal_pred_min, sim.yl_conformal, sim.yu_conformal, option="all"
)

qq_max = np.quantile(score_max, 0.95)
qq_min = np.quantile(score_min, 0.95)


def conformal_set(x_new, y_range, compute_score, qq):
    conformal_set = []
    for j in range(x_new.shape[0]):
        y_new_score = np.array(
            [compute_score(x_new[j], y_new, y_new) for y_new in y_range]
        )
        selected = np.where(y_new_score < qq)[0]
        conformal_set += [[y_new_range[selected].min(), y_new_range[selected].max()]]
    return np.array(conformal_set).T


conformal_set_max = sim.y_eval_pred.max(axis=0) + qq_max
conformal_set_min = sim.y_eval_pred.min(axis=0) - qq_min


# %%

# %%


def calculate_coverage_probability(
    y, sim, set_min: np.ndarray, set_max: np.ndarray
) -> np.ndarray:
    prob_coverage = np.zeros(y.shape[0])
    for j in range(y.shape[0]):
        y_random = y[j] + sim.gen_noise(N=10000)
        prob_coverage[j] = np.mean((y_random > set_min[j]) & (y_random < set_max[j]))
    return prob_coverage


# Calculate and plot the coverage probability for the first scenario
prob_coverage = calculate_coverage_probability(
    sim.y_eval_signal, sim, conformal_set[0], conformal_set[1]
)
plt.plot(
    sim.x_eval[:, -1],
    prob_coverage,
    label="Coverage of conformal sets",
    linestyle="-",
    linewidth=2,
)

# Calculate and plot the coverage probability for the second scenario
prob_coverage = calculate_coverage_probability(
    sim.y_eval_signal, sim, y_hat - qq2, y_hat + qq2
)
plt.plot(
    sim.x_eval[:, -1],
    prob_coverage,
    label="Coverage of conformal set(y observed)",
    linestyle="--",
    linewidth=2,
)

# Calculate and plot the coverage probability for the third scenario
prob_coverage = calculate_coverage_probability(
    sim.y_eval_signal,
    sim,
    sim.y_eval_signal - 1.96 * (sim.std_eps),
    sim.y_eval_signal + 1.96 * (sim.std_eps),
)
plt.plot(
    sim.x_eval[:, -1],
    prob_coverage,
    label="Coverage of y_signal +- 1.96*sigma_eps",
    linestyle="-.",
    linewidth=2,
)

# Add a title and labels for the x and y axes
plt.title("Coverage Probability")
plt.xlabel("x_eval")
plt.ylabel("Probability")

# Adjust the position of the legend
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Show the plot
plt.show()
# %%
# Calculate the widths
width = conformal_set[1] - conformal_set[0]
width2 = 2 * qq2 * np.ones(sim.x_eval.shape[0])
width3 = 2 * 1.96 * sim.std_eps * np.ones(sim.x_eval.shape[0])
width4 = conformal_set_max - conformal_set_min

# Plot each width
plt.plot(sim.x_eval[:, -1], width, label="Width 1")
plt.plot(sim.x_eval[:, -1], width2, label="Width 2")
plt.plot(sim.x_eval[:, -1], width3, label="Width 3")
plt.plot(sim.x_eval[:, -1], width4, label="Width 4")
plt.ylim(0, np.max(width) + 1)
# Add a legend
plt.legend()

# Show the plot
plt.show()
# %%
cc = conformal_score[sim.indices]

np.quantile(cc, 0.95, axis=1).max()
yy = np.linspace(-5, 15, 1000)

score = compute_score(sim.y_eval_pred[sim.indices], sim.y_eval_pre, option="all")
