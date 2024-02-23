# %%
import numpy as np
from utils_sim import Data


from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# %%


def get_intervals(y, scale):
    yl = np.floor(y / scale) * scale
    yu = np.ceil(y / scale) * scale
    return yl, yu


def split_data(data, train_ratio=0.5):
    (
        data.x_train,
        data.x_test,
        data.yl_train,
        data.yl_test,
        data.yu_train,
        data.yu_test,
        data.y_train,
        data.y_test,
    ) = train_test_split(
        data.x,
        data.yl,
        data.yu,
        data.y,
        test_size=train_ratio,
        random_state=42,
    )


def update(dict, value, key):
    dict[key] = value
    return value


class Data:
    def __init__(self, N1, N2, K, beta, pos, scale, gen_x, gen_noise, gen_y_signal):
        self.params = {}
        self.N1 = update(self.params, N1, "Sample size")
        self.N2 = update(self.params, N2, "Conformal sample size")
        self.K = update(self.params, K, "Feature dim")
        self.beta = update(self.params, beta, "Beta")
        self.pos = update(self.params, pos, "Position of non-zero beta")
        self.scale = update(self.params, scale, "Interval scale")
        self.gen_x = gen_x
        self.gen_noise = gen_noise
        self.gen_y_signal = gen_y_signal

    def generate_data(self):
        self.x = self.gen_x(N=self.N1, K=self.K)
        self.epsilon = self.gen_noise(N=self.N1)
        self.y = self.gen_y_signal(self.x, self.params) + self.epsilon
        self.yl, self.yu = self.get_intervals(self.y)

        self.x_conformal = self.gen_x(N=self.N2, K=self.K)
        self.epsilon_conformal = self.gen_noise(N=self.N2)
        self.y_conformal = (
            self.gen_y_signal(self.x_conformal, self.params) + self.epsilon_conformal
        )
        self.yl_conformal, self.yu_conformal = self.get_intervals(self.y_conformal)

        self.x_eval, self.y_eval_signal = self.gen_eval()
        self.yl_eval, self.yu_eval = self.get_intervals(self.y_eval_signal)


sim = Data(
    N1=400,
    N2=200,
    K=1000,
    beta=np.ones(10),
    pos=[0, 1, 2, 3, 4],
    scale=1,
    gen_x=lambda N, K: np.random.uniform(-np.sqrt(3), np.sqrt(3), (N, K)),
    gen_noise=lambda N: np.random.normal(0, 1, N),
    gen_y_signal=lambda x, beta, pos: x[:, pos] @ beta[pos],
)

sim.x, sim.x_conformal = map(sim.gen_x, [sim.N1, sim.N2], [sim.K, sim.K])
sim.y_signal, sim.y_conformal_signal = map(
    sim.gen_y_signal, [sim.x, sim.x_conformal], [sim.beta, sim.beta], [sim.pos, sim.pos]
)

sim.y = sim.y_signal + sim.gen_noise(sim.N1)
sim.y_conformal = sim.y_conformal_signal + sim.gen_noise(sim.N2)

(sim.yl, sim.yu), (sim.yl_conformal, sim.yu_conformal) = map(
    get_intervals, [sim.y, sim.y_conformal], [sim.scale, sim.scale]
)

(
    sim.x_train,
    sim.x_test,
    sim.yl_train,
    sim.yl_test,
    sim.yu_train,
    sim.yu_test,
    sim.y_train,
    sim.y_test,
) = train_test_split(sim.x, sim.yu, sim.yl, sim.y, test_size=0.5, random_state=42)

# %%
lasso_lower = Lasso(alpha=0.2).fit(sim.x, sim.yl_train)
lasso_upper = Lasso(alpha=0.2).fit(sim.x, sim.yu_train)
# %%
