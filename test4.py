# %%
import numpy as np
from utils_dgp import SimData
from wlpy.gist import current_time

def run_simulation(est_kwargs, sample_size=400):
    sim = SimData(N=sample_size)
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