# %%
from utils_sim import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from wlpy.gist import current_time

current_dt = current_time()

dgp_params = {
    "N": 2000,
    "K": 200,
    "eps_std": 1.0,
    "pos": [0, 1, 2, 3, 4],
    "scale": 4.0,
}

gen_eps = lambda N, eps_std, **kwargs: np.random.chisquare(3, N) - 3

data = Data(gen_y_signal=default_gen_y_signal, dgp_params=dgp_params)
# data = Data(gen_y_signal=gen_y_signal_2, dgp_params=dgp_params)
# %%
from sklearn.linear_model import Lasso, LinearRegression

#  Fit models to the training data
lasso_l = Lasso(alpha=0.2).fit(data.x_train, data.yl_train)
lasso_u = Lasso(alpha=0.2).fit(data.x_train, data.yu_train)
lasso_oracle = Lasso(alpha=0.2).fit(data.x_train, data.y_train)

linear_reg = LinearRegression().fit(data.x_train[:, :5], data.y_train)

krr_l = KernelRidge(kernel="rbf", alpha=2).fit(data.x_train, data.yl_train)
krr_u = KernelRidge(kernel="rbf", alpha=2).fit(data.x_train, data.yu_train)
krr_oracle = KernelRidge(alpha=2).fit(data.x_train, data.y_train)

# rf_l = RandomForestRegressor(n_estimators=200).fit(data.x_train, data.yl_train)
# rf_u = RandomForestRegressor(n_estimators=200).fit(data.x_train, data.yu_train)
# rf_oracle = RandomForestRegressor(n_estimators=200).fit(data.x_train, data.y_train)
print(score_func_sq(lasso_l.predict(data.x_eval), data.yl_reg).mean())
print(score_func_sq(krr_l.predict(data.x_eval), data.yl_reg).mean())
# print(score_func_sq(rf_l.predict(data.x_eval), data.yl_reg).mean())
# print(score_func_sq(lasso_oracle.predict(data.x_eval), data.y_eval_signal).mean())
print(score_func_sq(krr_oracle.predict(data.x_eval), data.y_eval_signal).mean())
# print(score_func_sq(rf_oracle.predict(data.x_eval), data.y_eval_signal).mean())


# %%
# test the empirical probability of y in the conformal set, by generating new data
y_new = data.y_eval_samples
conf_lasso = combined_conformal_intervals(
    data, lasso_l, lasso_u, lasso_oracle, score_func_abs_val
)
# conf_rf = combined_conformal_intervals(data, rf_l, rf_u, rf_oracle, score_func_abs_val)
np.quantile(score_func_abs_val(lasso_oracle.predict(data.x_test), data.y_test), 0.95)
# np.quantile(
#     score_func_abs_val(linear_reg.predict(data.x_test[:, :5]), data.y_test), 0.95
# )

# %%
conf = conf_lasso


empirical_prob_comb = calculate_proportion(y_new, conf["combined conformal intervals"])
empirical_prob_oracle = calculate_proportion(y_new, conf["oracle conformal intervals"])
empirical_prob_optimal = calculate_proportion(
    y_new, [data.y_eval_signal - 1.96, data.y_eval_signal + 1.96]
)


import matplotlib.pyplot as plt

plt.plot(empirical_prob_comb, "o", alpha=0.5, label="Conformal Set (Combined)")
plt.plot(empirical_prob_oracle, "o", alpha=0.5, label="Conformal Set (Oracle)")
plt.plot(empirical_prob_optimal, alpha=0.5, label="Optimal Set")
plt.legend()
plt.xlabel("X_* Index")
plt.ylabel("Empirical Coverage Probability")
plt.title("Empirical Probability Comparison")
# plt.savefig(f"simulation-results/{current_dt}_empirical_prob_comparison.pdf")
plt.show()

# (empirical_prob_oracle >= 0.95).mean()
(empirical_prob_comb >= 0.95).mean()

# %%
plt.plot(
    conf["combined conformal intervals"][1] - conf["combined conformal intervals"][0],
    alpha=0.5,
    label="Conformal Set (Combined)",
)
plt.plot(
    conf["oracle conformal intervals"][1] - conf["oracle conformal intervals"][0],
    alpha=0.5,
    label="Conformal Set (Oracle)",
)
plt.plot(
    data.y_eval_signal + 1.96 - (data.y_eval_signal - 1.96),
    alpha=0.5,
    label="Optimal Set",
)
plt.legend()
plt.xlabel("X_* Index")
plt.ylabel("Interval Width")
plt.title("Interval Width Comparison")
plt.ylim(0, 20)
# plt.savefig(f"simulation-results/{current_dt}_interval_width_comparison.pdf")
print(
    np.max(
        conf["combined conformal intervals"][1]
        - conf["combined conformal intervals"][0]
    )
)


# %%
def score_brackets(yl, yu, yl_pred, yu_pred):
    # return np.maximum(0,  yu - yu_pred) ** 2 + np.maximum(0, yl_pred- yl)**2
    # return np.maximum(
    #         np.maximum(0, yu - yu_pred) ** 2, np.maximum(0, yl_pred - yl) ** 2
    #     )
    # return (yu - yu_pred) ** 2 + (yl_pred - yl) ** 2
    return np.abs(yu - yu_pred) + np.abs(yl_pred - yl)
    # return np.maximum((yu - yu_pred) ** 2, (yl_pred - yl) ** 2)
    # return np.maximum(yu_pred-yu,  yl-yl_pred)


ss = score_brackets(
    data.yl_test,
    data.yu_test,
    lasso_l.predict(data.x_test),
    lasso_u.predict(data.x_test),
)

qq = np.quantile(ss, [0.95])
print(f"score: {ss}")
plt.plot(np.arange(len(ss)), ss)
print(f"Quantile: {qq}")
data.x_train.shape
# given y is in range(-20, 20): for each x in data.x_eval, search for the interval that contains y such that score_sq_brackets(y, y, lasso_l.predict(data.x_eval), lasso_u.predict(data.x_eval)) <= qq
# %%
intervals = []
score_save = np.zeros((len(data.x_eval), 200))
i, j = 0, 0
yrange = np.arange(-10, 10, 0.1)
for x in data.x_eval:
    pred_l = lasso_l.predict([x])
    pred_u = lasso_u.predict([x])
    lower_bound = None
    upper_bound = None
    for y in yrange:
        score = score_brackets(y, y, pred_l, pred_u)
        if score <= qq:
            if lower_bound is None:
                lower_bound = y
            upper_bound = y
        score_save[i, j] = score
        j += 1
    j = 0
    i += 1
    intervals.append((lower_bound, upper_bound))
intervals = np.array(intervals).T
plt.plot(yrange, score_save[2, :])
# %%
empirical_prob_sq = calculate_proportion(y_new, intervals)

plt.plot(empirical_prob_sq, "o", alpha=0.5, label="Squared Score")
plt.plot(empirical_prob_comb, "o", alpha=0.5, label="Conformal Set (Combined)")
plt.plot(empirical_prob_oracle, "o", alpha=0.5, label="Conformal Set (Oracle)")
plt.plot(empirical_prob_optimal, alpha=0.5, label="Optimal Set")
plt.legend()
plt.xlabel("X_* Index")
plt.ylabel("Empirical Coverage Probability")
plt.title("Empirical Probability Comparison")
# plt.savefig(f"simulation-results/{current_dt}_empirical_prob_comparison.pdf")
plt.show()
(empirical_prob_sq >= 0.95).mean()


lasso_l.predict(data.x_eval)[0]
lasso_u.predict(data.x_eval)[0]
# %%
plt.plot(
    conf["combined conformal intervals"][1] - conf["combined conformal intervals"][0],
    alpha=0.5,
    label="Conformal Set (Combined)",
)
plt.plot(
    conf["oracle conformal intervals"][1] - conf["oracle conformal intervals"][0],
    alpha=0.5,
    label="Conformal Set (Oracle)",
)
plt.plot(
    data.y_eval_signal + 1.96 - (data.y_eval_signal - 1.96),
    alpha=0.5,
    label="Optimal Set",
)
plt.plot(intervals[1] - intervals[0], alpha=0.5, label="Alternative Score")
plt.legend()
plt.xlabel("X_* Index")
plt.ylabel("Interval Width")
plt.title("Interval Width Comparison")
plt.ylim(0, 20)
# plt.savefig(f"simulation-results/{current_dt}_interval_width.pdf")
plt.show()
# %%
data.yl_sample.mean(axis=0)

intervals = np.quantile(data.yl_sample, [0.025, 0.975], axis=0)
intervals[1] = intervals[1] + dgp_params["scale"]
# %%
