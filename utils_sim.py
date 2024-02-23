# %%
import numpy as np
from sklearn.model_selection import train_test_split


default_dgp_params = {
    "N": 1000,
    "K": 200,
    "eps_std": 1,
    "pos": [1, 2, 3, 4, 5],
    "scale": 4.0,
}

default_gen_x = lambda N, K, **kwargs: np.random.uniform(-2, 2, [N, K])
default_gen_eps = lambda N, eps_std, **kwargs: np.random.normal(0, eps_std, N)
default_gen_y_signal = lambda x, pos, **kwargs: np.inner(x[:, pos], np.ones(len(pos)))
default_get_interval = lambda y, scale, **kwargs: (
    np.floor(y / scale) * scale,
    np.ceil(y / scale) * scale,
)


def calculate_proportion(samples, bounds):
    lower_bound, upper_bound = bounds
    within_bounds = np.logical_and(samples >= lower_bound, samples <= upper_bound)
    proportion_within_bounds = np.mean(within_bounds, axis=0)
    return proportion_within_bounds


class Data:
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

        self.x_eval = self.gen_x(N=1000, K=self.K)
        self.x_eval = self.x_eval[
            np.argsort(np.linalg.norm(self.x_eval[:, self.pos], axis=1))
        ]

        self.y_eval_signal = self.gen_y_signal(self.x_eval, **dgp_params)
        self.y_eval_samples = self.y_eval_signal + self.gen_eps(
            N=[2000, self.y_eval_signal.shape[0]], eps_std=self.eps_std
        )

        self.yl_sample, self.yu_sample = self.get_interval(
            self.y_eval_samples, **self.dgp_params
        )
        self.yl_reg = self.yl_sample.mean(axis=0)
        self.yu_reg = self.yu_sample.mean(axis=0)


score_func_abs_val = lambda y_obs, y_pred: np.abs(y_obs - y_pred)
score_func_sq = lambda y_obs, y_pred: (y_obs - y_pred) ** 2

gen_y_signal_2 = (
    lambda x, **kwargs: 2 * x[:, 0] ** 2
    + np.sin(x[:, 1])
    + np.sin(x[:, 2])
    + x[:, 3] ** 3
    + x[:, 4] ** 3
)


def split_conformal_inference_abs_score(
    x_calib, y_calib, predictor, score_func, x_eval
):
    qq = score_quantile(x_calib, y_calib, predictor, score_func)

    y_new_pred = predictor(x_eval)

    def interval(pred, qq):
        return np.array([pred - qq, pred + qq])

    return interval(y_new_pred, qq)


def score_quantile(x_test, y_test, predictor, score_func):
    y_test_pred = predictor(x_test)
    score = score_func(y_test, y_test_pred)
    qq = np.quantile(score, [0.95])
    return qq


def combined_conformal_intervals(
    data, predictor_l, predictor_u, predictor_oracle, score_func
):
    conformal_set_l = split_conformal_inference_abs_score(
        data.x_test,
        data.yl_test,
        predictor_l.predict,
        score_func=score_func,
        x_eval=data.x_eval,
    )
    conformal_set_u = split_conformal_inference_abs_score(
        data.x_test,
        data.yu_test,
        predictor_u.predict,
        score_func=score_func,
        x_eval=data.x_eval,
    )
    return {
        "combined conformal intervals": np.array(
            [conformal_set_l[0], conformal_set_u[1]]
        ),
        "oracle conformal intervals": split_conformal_inference_abs_score(
            data.x_test,
            data.y_test,
            predictor_oracle.predict,
            score_func=score_func,
            x_eval=data.x_eval,
        ),
    }
