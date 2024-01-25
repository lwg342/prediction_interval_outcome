import numpy as np
from sklearn.model_selection import train_test_split
from typing import Callable, List, Optional, Tuple


def compute_score(y_test_pred, yu_test, yl_test, option="mean"):
    diff_l = np.maximum(yl_test - y_test_pred, 0) ** 2
    diff_u = np.maximum(y_test_pred - yu_test, 0) ** 2
    if option == "mean":
        return (diff_l + diff_u).mean(axis=1)
    if option == "all":
        return diff_l + diff_u


class SimData:
    """
    A class representing simulated data for ensemble interval prediction.

    Parameters:
    - N1 (int): Number of training samples.
    - N2 (int): Number of validation samples.
    - M (int): Number of ensemble members.
    - K (int): Number of features.
    - params (float): Parameter value.
    - x_distri (str): Distribution of input features.
    - epsilon_distri (str): Distribution of noise.
    - var_epsilon (float): Variance of noise.
    - df (int): Degrees of freedom for noise distribution.
    - interval_bias (Optional[List[float]]): Bias for interval calculation.
    - scale (float): Scaling factor for interval calculation.
    - n_test_points (int): Number of test points.
    - cal_y_signal (Optional[Callable]): Function to calculate y signal.
    - Beta_params (Optional[List[int]]): Parameters for Beta distribution.

    Attributes:
    - N1 (int): Number of training samples.
    - N2 (int): Number of validation samples.
    - M (int): Number of ensemble members.
    - K (int): Number of features.
    - params (float): Parameter value.
    - var_epsilon (float): Variance of noise.
    - scale (float): Scaling factor for interval calculation.
    - interval_bias (Optional[List[float]]): Bias for interval calculation.
    - x_distri (str): Distribution of input features.
    - epsilon_distri (str): Distribution of noise.
    - df (int): Degrees of freedom for noise distribution.
    - n_test_points (int): Number of test points.
    - Beta_params (Optional[List[int]]): Parameters for Beta distribution.

    Methods:
    - calculate_weights_and_yd(Beta_params: Optional[List[int]] = None, M: Optional[int] = None) -> None:
        Calculates weights and yd values based on Beta distribution parameters and M value.
    - get_interval(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Calculates the lower and upper bounds of the interval based on the given y values.
    - gen_x(N: int, K: int) -> np.ndarray:
        Generates input features x with the specified number of samples and features.
    - gen_noise(N: int) -> np.ndarray:
        Generates noise epsilon with the specified number of samples.
    """

    def __init__(
        self,
        N1: int = 1000,
        N2: int = 1000,
        M: int = 200,
        K: int = 4,
        params: float = np.pi / 10,
        x_distri: str = "uniform",
        epsilon_distri: str = "normal",
        std_eps: float = 1.0,
        df: int = 2,
        interval_bias: Optional[List[float]] = None,
        scale: float = 1.0,
        n_test_points: int = 100,
        cal_y_signal: Optional[Callable] = None,
        Beta_params: Optional[List[int]] = None,
    ):
        self.N1 = N1
        self.N2 = N2
        self.M = M
        self.K = K
        self.params = params

        self.std_eps = std_eps
        self.scale = scale
        self.interval_bias = interval_bias if interval_bias is not None else [0.0, 0.0]
        self.x_distri = x_distri
        self.epsilon_distri = epsilon_distri
        self.df = df
        self.n_test_points = n_test_points
        self.Beta_params = Beta_params if Beta_params is not None else [1, 1]
        if cal_y_signal is None:
            self.cal_y_signal = (
                lambda x, params: np.sqrt(2)
                + np.dot(x, params * np.ones(self.K))
                + np.cos(x[:, -1] * 3)
                + 0.5 * x[:, -1] ** 2
            )
        else:
            self.cal_y_signal = cal_y_signal

    def generate_data(self):
        self.x = self.gen_x(N=self.N1, K=self.K)
        self.epsilon = self.gen_noise(N=self.N1)
        self.y = self.cal_y_signal(self.x, self.params) + self.epsilon
        self.yl, self.yu = self.get_intervals(self.y)

        self.x_conformal = self.gen_x(N=self.N2, K=self.K)
        self.epsilon_conformal = self.gen_noise(N=self.N2)
        self.y_conformal = (
            self.cal_y_signal(self.x_conformal, self.params) + self.epsilon_conformal
        )
        self.yl_conformal, self.yu_conformal = self.get_intervals(self.y_conformal)

        self.x_eval, self.y_eval_signal = self.gen_eval()
        self.yl_eval, self.yu_eval = self.get_intervals(self.y_eval_signal)

    @property
    def y_middle(self):
        return (self.yl + self.yu) / 2

    def calculate_weights_and_yd(self, weights=None, Beta_params=None, M=None):
        if Beta_params == None:
            Beta_params = self.Beta_params
        if M is None:
            M = self.M
        if weights is None:
            self.weights = np.random.beta(Beta_params[0], Beta_params[1], [M, self.N1])
        else:
            self.weights = weights
        self.yd = self.weights * self.yl + (1 - self.weights) * self.yu

    def split_data(self, train_ratio=0.5):
        (
            self.x_train,
            self.x_test,
            self.yu_train,
            self.yu_test,
            self.yl_train,
            self.yl_test,
            self.y_train,
            self.y_test,
            self.yd_train,
            self.yd_test,
        ) = train_test_split(
            self.x,
            self.yu,
            self.yl,
            self.y,
            self.yd.T,
            test_size=train_ratio,
            random_state=42,
        )

    def fit(self, **est_kwargs):
        self.models = {
            target: fit_model(self.x, getattr(self, target), **est_kwargs)
            for target in ["y", "yl", "yu"]
        }
        self.models["yd_train"] = fit_model(self.x_train, self.yd_train, **est_kwargs)

        self.y_eval_pred = self.models["yd_train"].predict(self.x_eval).T
        self.y_test_pred = self.models["yd_train"].predict(self.x_test).T
        self.y_conformal_pred = self.models["yd_train"].predict(self.x_conformal).T

        self.y_eval_pred_obs = self.models["y"].predict(self.x_eval)
        self.y_test_pred_obs = self.models["y"].predict(self.x_test)
        self.y_conformal_pred_obs = self.models["y"].predict(self.x_conformal)

        self.tolerance = est_kwargs["tolerance"]
        self.score = compute_score(self.y_test_pred, self.yu_test, self.yl_test)
        self.indices = select_indices(self.score, self.tolerance)
        self.res_min, self.res_max = pred_error(
            self.y_eval_signal, self.y_eval_pred[self.indices]
        )

    def get_intervals(self, y):
        """
        Get the lower and upper interval values for the given y values.

        Args:
            y (numpy.ndarray): Array of y values.

        Returns:
            numpy.ndarray: Array of lower interval values.
            numpy.ndarray: Array of upper interval values.
        """
        yl = np.floor(y / self.scale) * self.scale - self.interval_bias[0]
        yu = np.ceil(y / self.scale) * self.scale + self.interval_bias[1]
        return yl, yu

    def gen_x(self, N: int, K: int) -> np.ndarray:
        if self.x_distri == "normal":
            x = np.random.normal(0, 1, (N, K))
        elif self.x_distri == "uniform":
            x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (N, K)) 
        return x

    def gen_noise(self, N: int) -> np.ndarray:
        if self.epsilon_distri == "normal":
            epsilon = np.random.normal(0, self.std_eps, N)
        elif self.epsilon_distri == "chisquare":
            epsilon = (np.random.chisquare(self.df, N) - self.df) / np.sqrt(2 * self.df)
        elif self.epsilon_distri == "no_noise":
            epsilon = np.zeros(N)
        return epsilon

    def gen_eval(self):
        x_eval = np.column_stack(
            (
                np.ones((self.n_test_points, self.K - 1)) * 0,
                np.linspace(
                    np.min(self.x[:, -1]), np.max(self.x[:, -1]), self.n_test_points
                ),
            )
        )
        y_eval_signal = self.cal_y_signal(x_eval, self.params)
        return x_eval, y_eval_signal

    def __repr__(self):
        params_dict = {
            "N1 training and testing": self.N1,
            "N2 conformal sample": self.N2,
            "M random draws from the interval": self.M,
            "K num of features": self.K,
            "Beta params": self.Beta_params,
            "var_epsilon": self.std_eps,
            "interval_bias": self.interval_bias,
            "x_distri": self.x_distri,
            "epsilon_distri": self.epsilon_distri,
            "df": self.df,
            "tolerance": self.tolerance,
            "n_test_points": self.n_test_points,
        }
        params_strs = [f"{k}: {v}" for k, v in params_dict.items()]
        return "\n".join(params_strs)


from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


def select_indices(score, tolerance):
    """
    Selects the indices of predictions that have a score within a given tolerance.

    Parameters:
    - compute_score (function): A function that computes the score of predictions.
    - tolerance (float): The tolerance value to determine the acceptable range of scores.
    - yu_test (array-like): The upper bounds of the true values.
    - yl_test (array-like): The lower bounds of the true values.
    - y_test_pred (array-like): The predicted values.

    Returns:
    - indices (array-like): The indices of predictions that have a score within the tolerance range.
    """
    smallest_score = np.min(score)
    threshold = smallest_score + tolerance
    indices = np.where(score <= threshold)[0]
    return indices


def pred_error(y_true, y_pred):
    res_min = np.min(np.abs(y_true - y_pred), axis=0)
    res_max = np.max(np.abs(y_true - y_pred), axis=0)
    return res_min, res_max


def get_loclin_pred(eval, loclin_models):
    pred = np.column_stack([mm.vec_fit(eval) for mm in loclin_models]).T
    return pred


def fit_model(x, y, method, **kwargs):
    if method == "linear":
        model = LinearRegression()
    elif method == "rf":
        model = RandomForestRegressor(
            n_estimators=kwargs["rf_n_estimators"], max_depth=kwargs["rf_max_depth"]
        )
    elif method == "krr":
        model = KernelRidge(alpha=kwargs["krr_alpha"], kernel=kwargs["krr_kernel"])

    model = model.fit(x, y)
    return model


def calculate_coverage_probability(
    y, sim, set_min: np.ndarray, set_max: np.ndarray
) -> np.ndarray:
    prob_coverage = np.zeros(y.shape[0])
    for j in range(y.shape[0]):
        y_random = y[j] + sim.gen_noise(N=10000)
        prob_coverage[j] = np.mean((y_random >= set_min[j]) & (y_random <=  set_max[j]))
    return prob_coverage
