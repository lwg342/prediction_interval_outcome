import numpy as np
from sklearn.model_selection import train_test_split

default_dgp_params = {
    "N": 1000,
    "K": 1,
    "eps_std": 1.0,
    "scale": 1.0,
}

default_gen_x = lambda N, K, **kwargs: np.random.uniform(-2, 2, [N, K])
default_gen_eps = lambda N, eps_std, **kwargs: np.random.normal(0, eps_std, N)
default_gen_y_signal = lambda x, **kwargs: np.inner(x, np.ones(x.shape[1]))
default_get_interval = lambda y, scale, **kwargs: (
    np.floor(y / scale) * scale,
    np.ceil(y / scale) * scale,
)


class SimData:
    def __init__(
        self,
        dgp_params=default_dgp_params,
        gen_x=default_gen_x,
        gen_eps=default_gen_eps,
        gen_y_signal=default_gen_y_signal,
        get_interval=default_get_interval,
    ):
        self.dgp_params = dgp_params
        self.gen_x = gen_x
        self.gen_eps = gen_eps
        self.gen_y_signal = gen_y_signal
        self.get_interval = get_interval

        for key, value in dgp_params.items():
            setattr(self, key, value)

        self.x = self.gen_x(**dgp_params)
        self.eps = self.gen_eps(**dgp_params)
        self.y_signal = self.gen_y_signal(self.x, **dgp_params)
        self.y = self.y_signal + self.eps
        self.yl, self.yu = self.get_interval(self.y, **dgp_params)

        (
            self.x_train,
            self.x_test,
            self.yl_train,
            self.yl_test,
            self.yu_train,
            self.yu_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(self.x, self.yl, self.yu, self.y)
        print("DGP params:", dgp_params)
        # Compute the evaluation set

    def gen_eval(self, n_eval=100, sample_size=5000):
        self.x_eval = np.linspace(self.x.min(), self.x.max(), n_eval).reshape(-1, 1)
        self.y_eval_signal = self.gen_y_signal(self.x_eval, **self.dgp_params)
        self.eps_eval_samples = self.gen_eps(
            N=[sample_size, self.y_eval_signal.shape[0]], eps_std=self.eps_std
        )
        self.y_eval_samples = self.y_eval_signal + self.eps_eval_samples
        self.yl_eval_samples, self.yu_eval_samples = self.get_interval(
            self.y_eval_samples, **self.dgp_params
        )

        self.yl_reg = self.yl_eval_samples.mean(axis=0)
        self.yu_reg = self.yu_eval_samples.mean(axis=0)


class IntervalCensoredData:
    def __init__(self, x, y, yl, yu):
        self.x = x
        self.y = y
        self.yl = yl
        self.yu = yu
        (
            self.x_train,
            self.x_test,
            self.yl_train,
            self.yl_test,
            self.yu_train,
            self.yu_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(self.x, self.yl, self.yu, self.y)
