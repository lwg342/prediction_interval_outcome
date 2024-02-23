# %%
import numpy as np
from utils_dgp import SimData, compute_score, calculate_coverage_probability
from wlpy.gist import current_time
import numpy as np

# %%


def lin_model(x, params):
    return 10 * np.ones(x.shape[0]) + 1 * x[:, -1]


sample_size = 2000
sim = SimData(
    N1=sample_size,
    scale=1.0,
    epsilon_distri="normal",
    std_eps=1,
    # cal_y_signal=lin_model,
)
sim.generate_data()
sim.calculate_weights_and_yd()
sim.split_data()

est_kwargs = {
    "method": "krr",
    "krr_kernel": "rbf",
    "krr_alpha": 0.02,
    "rf_max_depth": 10,
    "rf_n_estimators": 200,
    "tolerance": 1 / np.sqrt(sample_size),
}
sim.fit(**est_kwargs)
# %% Conformal inference part

conformal_score = compute_score(
    sim.y_conformal_pred, sim.yu_conformal, sim.yl_conformal, option="all"
)

conformal_score = conformal_score[sim.indices].max(axis=0)


y_new_range = np.linspace(-5, 15, 1000)
qq = np.quantile(conformal_score, 0.95)
print(f"qq: {qq}")
conformal_set = []
y_new_pred = sim.y_eval_pred[sim.indices]
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

score_if_y_observed = np.abs(sim.y_conformal - sim.y_conformal_pred_obs)
qq_obs = np.quantile(score_if_y_observed, 0.95)

y_conformal_pred_max = sim.y_conformal_pred[sim.indices].max(axis=0)
y_conformal_pred_min = sim.y_conformal_pred[sim.indices].min(axis=0)

score_max = compute_score(
    y_conformal_pred_max, sim.yu_conformal, sim.yl_conformal, option="all"
)
score_min = compute_score(
    y_conformal_pred_min, sim.yu_conformal, sim.yl_conformal, option="all"
)

qq_max = np.quantile(score_max, 0.95)
qq_min = np.quantile(score_min, 0.95)


conformal_set_max = sim.y_eval_pred.max(axis=0) + qq_max
conformal_set_min = sim.y_eval_pred.min(axis=0) - qq_min


import matplotlib.pyplot as plt

plt.scatter(sim.x_eval[:, -1], sim.res_min, s=0.5, alpha=0.5, label="res_min")
plt.scatter(sim.x_eval[:, -1], sim.res_max, s=0.5, alpha=0.5, label="res_max")
# plt.scatter(sim.x_eval[:, -1], sim.yl_eval)
# plt.scatter(sim.x_eval[:, -1], sim.yu_eval)

plt.plot(sim.x_eval[:, -1], sim.y_eval_signal, linewidth=2.5, label="true signal")
plt.fill_between(
    sim.x_eval[:, -1],
    sim.y_eval_pred.min(axis=0),
    sim.y_eval_pred.max(axis=0),
    alpha=1.0,
    label="pre-selection prediction set",
    hatch="///",
)
plt.fill_between(
    sim.x_eval[:, -1],
    sim.y_eval_pred[sim.indices].min(axis=0),
    sim.y_eval_pred[sim.indices].max(axis=0),
    # color="tab:red",
    alpha=0.5,
    label="post-selection prediction set",
)
plt.plot(
    sim.x_eval[:, -1],
    sim.y_eval_pred_obs,
    linestyle="dashed",
    label="prediction when we observe y",
)
plt.fill_between(sim.x_eval[:, -1], sim.models["yl"].predict(sim.x_eval),  sim.models["yu"].predict(sim.x_eval), alpha=0.5, label="prediction interval")
# plt.fill_between(
#     sim.x_eval[:, -1],
#     conformal_set[0],
#     conformal_set[1],
#     alpha=0.4,
#     label="conformal set",
# )
# plt.fill_between(
#     sim.x_eval[:, -1],
#     conformal_set_min,
#     conformal_set_max,
#     alpha=0.5,
#     label="conformal set min/max",
# )
# plt.fill_between(
#     sim.x_eval[:, -1],
#     sim.y_eval_pred_obs - qq_obs,
#     sim.y_eval_pred_obs + qq_obs,
#     alpha=0.4,
#     label="conformal set when observe y",
# )
# plt.fill_between(
#     sim.x_eval[:, -1],
#     sim.y_eval_signal - 1.96 * sim.std_eps,
#     sim.y_eval_signal + 1.96 * sim.std_eps,
#     alpha=0.4,
#     label="conformal set y_signal -+ 1.96",
# )

plt.title(
    f"$N$={sim.N1}, $M$={sim.M}, $K$={sim.K}, $v_\epsilon$={sim.std_eps}, $b_1$={sim.interval_bias[0]}, $b_2$={sim.interval_bias[1]}, \n $x$_distri={sim.x_distri}, $\epsilon$_distri={sim.epsilon_distri}, df={sim.df}, scale = {sim.scale}"
)
plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.show()
# plt.savefig(f"simulation-results/{current_time()}-pred_error.pdf", bbox_inches="tight")
# %% Coverage probability

# Calculate and plot the coverage probability for the first scenario
prob_coverage = calculate_coverage_probability(
    sim.y_eval_signal, sim, conformal_set[0], conformal_set[1]
)
plt.scatter(
    sim.x_eval[:, -1],
    prob_coverage,
    label="Coverage of conformal sets",
    alpha=0.5,
)

# Calculate and plot the coverage probability for the second scenario
prob_coverage = calculate_coverage_probability(
    sim.y_eval_signal, sim, sim.y_eval_pred_obs - qq_obs, sim.y_eval_pred_obs + qq_obs
)
plt.scatter(
    sim.x_eval[:, -1],
    prob_coverage,
    label="Coverage of conformal set(y observed)",
    alpha=0.5,
)

# Calculate and plot the coverage probability for the third scenario
prob_coverage = calculate_coverage_probability(
    sim.y_eval_signal,
    sim,
    sim.y_eval_signal - 1.96 * (sim.std_eps),
    sim.y_eval_signal + 1.96 * (sim.std_eps),
)
plt.scatter(
    sim.x_eval[:, -1],
    prob_coverage,
    label="Coverage of y_signal +- 1.96*sigma_eps",
    alpha=0.5,
)
plt.hlines(0.95, sim.x_eval[:, -1].min(), sim.x_eval[:, -1].max())
# Add a title and labels for the x and y axes
plt.title("Coverage Probability, Pr(y in the sets)")
plt.xlabel("x_eval")
plt.ylabel("Probability")
plt.ylim(0.8, 1)
# Adjust the position of the legend
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Show the plot
plt.show()
# %%
# Calculate the widths
width = conformal_set[1] - conformal_set[0]
width2 = 2 * qq_obs * np.ones(sim.x_eval.shape[0])
width3 = 2 * 1.96 * sim.std_eps * np.ones(sim.x_eval.shape[0])


# Plot each width
plt.plot(sim.x_eval[:, -1], width, label="Width 1")
plt.plot(sim.x_eval[:, -1], width2, label="Width 2")
plt.plot(sim.x_eval[:, -1], width3, label="Width 3")
plt.ylim(0, np.max(width) + 1)
# Add a legend
plt.legend()
plt.show()

# %%
