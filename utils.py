# %%
import numpy as np
from sklearn.model_selection import train_test_split


default_dgp_params = {
    "N": 1000,
    "K": 1,
    "eps_std": 1.0,
    "pos": [0],
    "scale": 1.0,
}

default_gen_x = lambda N, K, **kwargs: np.random.uniform(-2, 2, [N, K])
default_gen_eps = lambda N, eps_std, **kwargs: np.random.normal(0, eps_std, N)
default_gen_y_signal = lambda x, pos, **kwargs: np.inner(x[:, pos], np.ones(len(pos)))
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

        # print("DGP params:", dgp_p    arams)


class EmpiricalData:
    def __init__(
        self,
        df,
        x_cols=["Education", "Experience"],
        yl_col="Log_lower_bound",
        yu_col="Log_upper_bound",
    ):
        self.x = df[x_cols].to_numpy()
        self.yl = df[yl_col].to_numpy()
        self.yu = df[yu_col].to_numpy()
        (
            self.x_train,
            self.x_test,
            self.yl_train,
            self.yl_test,
            self.yu_train,
            self.yu_test,
        ) = train_test_split(df[x_cols], df[yl_col], df[yu_col])
        df.loc[:, "is_test"] = df.index.isin(self.x_test.index).copy()
        self.df = df

    def create_local_test_samples(self, local_condition):
        self.local_condition = local_condition

        for j, condition_j in enumerate(local_condition):
            where_j = condition_j(self.x_test)
            self.x_local_test[j] = self.x_test[where_j]
            self.yl_local_test[j] = self.yl_test[where_j]
            self.yu_local_test[j] = self.yu_test[where_j]


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


def score_quantile(x_test, y_test, predictor, score_func, alpha=0.05):
    y_test_pred = predictor(x_test)
    score = score_func(y_test, y_test_pred)
    qq = np.quantile(score, 1 - alpha)
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


# %%


def epanechnikov_kernel(u):
    k = 0.75 * (1 - u**2)
    return np.where(np.abs(u) <= 1, k, 0)


def multivariate_epanechnikov_kernel(x, xi, h):
    # x is the query point, xi is a point from data.x, and h is the bandwidth.
    # Scale distances in each dimension
    u = (x - xi) / h
    # Compute the product of the Epanechnikov kernel for each dimension
    kernel_values = epanechnikov_kernel(u)
    return np.prod(kernel_values, axis=1)


def compute_weights(weights_unnormalized):
    weights = weights_unnormalized / np.sum(weights_unnormalized)
    return weights


def silvermans_rule(x):
    """
    Compute Silverman's rule of thumb for bandwidth selection in the multivariate case.

    Parameters:
    - data_x: numpy array of shape (N, d) where N is the number of samples and d is the dimensionality

    Returns:
    - h: the estimated bandwidth
    """
    N, d = x.shape
    sigma = np.std(x, axis=0, ddof=1)  # Sample standard deviation for each dimension
    h = (4 / ((d + 2) * N)) ** (1 / (d + 4)) * np.mean(sigma)
    return h


def compute_weight_sum(yl, yu, weights, t0, t1):
    # Create a boolean mask for the condition
    mask = (t0 <= yl) & (yu <= t1)

    # Compute the sum of weights for elements satisfying the condition
    weight_sum = np.sum(weights[mask])

    return weight_sum


def eligible_t0t1(yl, yu, weights):
    positive_weights_indices = weights > 0
    t0 = np.sort(yl[positive_weights_indices])
    t1 = np.sort(yu[positive_weights_indices])[::-1]
    return t0, t1


def find_t_hat(yl_train, yu_train, compute_weight_sum, weights, t0c, t1c, alpha=0.05):
    interval_width = np.inf
    optimal_t0, optimal_t1 = None, None
    for t0_i in t0c:
        for t1_i in t1c:
            if t0_i < t1_i:
                weight_sum = compute_weight_sum(
                    yl_train[weights > 0],
                    yu_train[weights > 0],
                    weights[weights > 0],
                    t0_i,
                    t1_i,
                )
                if weight_sum >= 1 - alpha and t1_i - t0_i < interval_width:
                    interval_width = t1_i - t0_i
                    optimal_t0, optimal_t1 = t0_i, t1_i
                if weight_sum < 1 - alpha:
                    break
    pred_interval = [optimal_t0, optimal_t1]
    return pred_interval


def pred_interval(x, x_train, yl_train, yu_train, h, alpha=0.05):
    pred_interval = np.zeros([2, x.shape[0]])
    for i, x_i in enumerate(x):
        weights = compute_weights(
            multivariate_epanechnikov_kernel(
                x_i,
                x_train,
                h=h,
            )
        )
        # print(weights)

        t0c, t1c = eligible_t0t1(yl_train, yu_train, weights)

        pred_interval[:, i] = find_t_hat(
            yl_train, yu_train, compute_weight_sum, weights, t0c, t1c, alpha=alpha
        )
    return pred_interval


def find_oracle_interval(intvs, n, alpha):
    sorted_intvs = intvs[np.argsort(intvs[:, 0])]
    i = 0
    optimal_width = np.inf
    while i <= n * alpha:
        # print(i)
        sorted_upper = np.sort(sorted_intvs[i:, 1])[::-1]
        # print(sorted_upper)

        t1 = sorted_upper[np.floor(n * alpha - i).astype(int)]

        width = t1 - sorted_intvs[i, 0]
        if width < optimal_width:
            optimal_width = width
            optimal_interval = [
                sorted_intvs[i, 0],
                t1,
            ]
        i += 1
    # print(optimal_interval)
    return optimal_interval


def calculate_proportion(samples, bounds):
    lower_bound, upper_bound = bounds
    within_bounds = np.logical_and(samples >= lower_bound, samples <= upper_bound)
    proportion_within_bounds = np.mean(within_bounds, axis=0)
    return proportion_within_bounds


def calculate_proportion_interval(samples_lower, samples_upper, bounds):
    lower_bound, upper_bound = bounds
    within_bounds = np.logical_and(
        samples_lower >= lower_bound, samples_upper <= upper_bound
    )
    proportion_within_bounds = np.mean(within_bounds, axis=0)
    return proportion_within_bounds


def create_interactive_terms(list1, list2):
    """
    This function takes two lists and returns a list of interactive terms.
    Each interactive term is a tuple consisting of one element from each list.

    Parameters:
    list1 (list): The first list of elements.
    list2 (list): The second list of elements.

    Returns:
    list: A list of tuples representing the interactive terms.
    """
    interactive_terms = [(item1, item2) for item1 in list1 for item2 in list2]
    return interactive_terms
