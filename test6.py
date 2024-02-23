# %% 
import numpy as np
from utils_dgp import SimData, compute_score

sim = SimData(N1=2000, scale=2, epsilon_distri="normal", std_eps=1.0)
sim.generate_data()
sim.calculate_weights_and_yd()
sim.split_data()

sample_size = 2000
est_kwargs = {
    "method": "krr",
    "krr_kernel": "rbf",
    "krr_alpha": 0.02,
    "rf_max_depth": 10,
    "rf_n_estimators": 200,
    "tolerance": 1 / np.sqrt(sample_size),
}
sim.fit(**est_kwargs)
# %%
def conformal_set(y_new, score, qq):
    conformal_set = [np.min(y_new[score <= qq], axis=1), np.max(y_new[score >= qq], axis=1)]
    return conformal_set
# %%
conformal_score = compute_score(
    sim.y_conformal_pred, sim.yl_conformal, sim.yu_conformal, option="all"
)
# %%
conformal_score
