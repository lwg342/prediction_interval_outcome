# %%
from utils_sim import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from wlpy.gist import current_time

current_dt = current_time()

dgp_params = {
    "N": 1000,
    "K": 200,
    "eps_std": 1,
    "pos": [1, 2, 3, 4, 5],
    "scale": 4.0,
}

data = Data(gen_y_signal=gen_y_signal_2, dgp_params=dgp_params)
# %%
from sklearn.linear_model import Lasso

lasso_l = Lasso(alpha=0.2).fit(data.x_train, data.yl_train)
lasso_u = Lasso(alpha=0.2).fit(data.x_train, data.yu_train)
lasso_oracle = Lasso(alpha=0.2).fit(data.x_train, data.y_train)

krr_l = KernelRidge(alpha=20).fit(data.x_train, data.yl_train)
krr_u = KernelRidge(alpha=20).fit(data.x_train, data.yu_train)
krr_oracle = KernelRidge(alpha=2).fit(data.x_train, data.y_train)

rf_l = RandomForestRegressor(n_estimators=200).fit(data.x_train, data.yl_train)
rf_u = RandomForestRegressor(n_estimators=200).fit(data.x_train, data.yu_train)
rf_oracle = RandomForestRegressor(n_estimators=200,max_depth=200).fit(data.x_train, data.y_train)

print(score_func_sq(lasso_l.predict(data.x_eval), data.yl_reg).mean())
print(score_func_sq(krr_l.predict(data.x_eval), data.yl_reg).mean())
print(score_func_sq(rf_l.predict(data.x_eval), data.yl_reg).mean())
print(score_func_sq(lasso_oracle.predict(data.x_eval), data.y_eval_signal).mean())
print(score_func_sq(krr_oracle.predict(data.x_eval), data.y_eval_signal).mean())
print(score_func_sq(rf_oracle.predict(data.x_eval), data.y_eval_signal).mean())


# %%
# test the empirical probability of y in the conformal set, by generating new data
y_new = data.y_eval_samples
conf_lasso = combined_conformal_intervals(
    data, lasso_l, lasso_u, lasso_oracle, score_func_abs_val
)
conf_rf = combined_conformal_intervals(data, rf_l, rf_u, rf_oracle, score_func_abs_val)


# %%
conf = conf_rf


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
plt.savefig(f"simulation-results/{current_dt}_empirical_prob_comparison.pdf")
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
plt.savefig(f"simulation-results/{current_dt}_interval_width_comparison.pdf")
print(
    np.max(
        conf["combined conformal intervals"][1]
        - conf["combined conformal intervals"][0]
    )
)
# %%
