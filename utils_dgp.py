import numpy as np
from sklearn.model_selection import train_test_split


class SimData:
    """
    A class representing a simulation.

    Attributes:
        N (int): Number of samples.
        M (int): Number of simulations.
        K (int): Number of features.
        beta (float): Coefficient value.
        x_distri (str): Distribution of x values.
        epsilon_distri (str): Distribution of noise values.
        var_epsilon (float): Variance of noise values.
        df (int): Degrees of freedom for noise distribution.
        interval_bias (list): Bias for interval values.
        scale (float): Scale factor for interval values.
        tolerance (float): Tolerance value.
        n_test_points (int): Number of test points.

    Methods:
        gen_x(): Generate x values based on the specified distribution.
        gen_noise(): Generate noise values based on the specified distribution.
        get_intervals(y): Get the lower and upper interval values for the given y values.
    """

    def __init__(
        self,
        N=1000,
        M=200,
        K=4,
        params=np.pi / 10,
        x_distri="uniform",
        epsilon_distri="normal",
        var_epsilon=1.0,
        df=2,
        interval_bias=[0.0, 0.0],
        scale=1.0,
        n_test_points=100,
        cal_y_signal=None,
        Beta_params=[1, 1],  # parameters for Beta distribution
    ):
        self.N = N
        self.M = M
        self.K = K
        self.params = params

        self.var_epsilon = var_epsilon
        self.scale = scale
        self.interval_bias = interval_bias
        self.x_distri = x_distri
        self.epsilon_distri = epsilon_distri
        self.df = df
        self.n_test_points = n_test_points
        self.Beta_params = Beta_params
        if cal_y_signal == None:
            self.cal_y_signal = (
                lambda x, params: np.sqrt(2)
                + np.dot(x, params * np.ones(self.K))
                + np.cos(x[:, -1] * 3)
                + 0.5 * x[:, -1] ** 2
            )

    def generate_data(self):
        self.x = self.gen_x()
        self.epsilon = self.gen_noise()
        self.y = self.cal_y_signal(self.x, self.params) + self.epsilon
        self.yl, self.yu = self.get_intervals(self.y)
        self.y_middle = (self.yl + self.yu) / 2
        self.x_eval, self.y_eval_signal = self.gen_eval()

    def calculate_weights_and_yd(self, Beta_params=None, M=None):
        if Beta_params == None:
            Beta_params = self.Beta_params
        if M == None:
            M = self.M
        self.weights = np.random.beta(Beta_params[0], Beta_params[1], [M, self.N])
        self.yd = self.weights * self.yl + (1 - self.weights) * self.yu

    def fit_and_predict(self, **estimation_kwargs):
        (
            self.y_test_pred,
            self.y_eval_pred,
            self.y_mid_fit,
            self.y_true_fit,
            self.yl_fit,
            self.yu_fit,
        ) = fit_and_predict(self, **estimation_kwargs)
        
        self.tolerance = estimation_kwargs["tolerance"]
        self.score = compute_score(self.y_test_pred, self.yu_test, self.yl_test)
        self.indices = select_indices(
            self.score, self.tolerance
        )

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

    def get_interval(self, y):
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

    def gen_x(self):
        """
        Generate x values based on the specified distribution.

        Returns:
            numpy.ndarray: Array of x values.
        """
        if self.x_distri == "normal":
            x = np.random.normal(0, 1, (self.N, self.K))
        if self.x_distri == "uniform":
            x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (self.N, self.K))
        return x

    def gen_noise(self):
        """
        Generate noise values based on the specified distribution.

        Returns:
            numpy.ndarray: Array of noise values.
        """
        if self.epsilon_distri == "normal":
            epsilon = np.random.normal(0, self.var_epsilon, self.N)
        if self.epsilon_distri == "chisquare":
            epsilon = (np.random.chisquare(self.df, self.N) - self.df) / np.sqrt(
                2 * self.df
            )
        if self.epsilon_distri == "no_noise":
            epsilon = np.zeros(self.N)
        return epsilon

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

    def create_param_dict(self):
        params_dict = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "beta": self.beta,
            "var_epsilon": self.var_epsilon,
            "interval_bias": self.interval_bias,
            "x_distri": self.x_distri,
            "epsilon_distri": self.epsilon_distri,
            "df": self.df,
            "tolerance": self.tolerance,
            "n_test_points": self.n_test_points,
            "x_eval": self.x_eval,
            "y_eval_signal": self.y_eval_signal,
        }
        return params_dict


from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from wlpy.regression import LocLin


def fit_and_predict(
    data,
    method="krr",
    krr_kernel="rbf",
    krr_alpha=0.1,
    rf_max_depth=10,
    rf_n_estimators=200,
    **kwargs
):
    if method not in ["loclin", "linear", "rf", "krr"]:
        raise ValueError("method must be one of 'loclin', 'linear' or 'rf'")
    if method == "loclin":
        model = [LocLin(data.x_train, j) for j in data.yd_train.T]
        y_test_pred = get_loclin_pred(data.x_test, model)
        y_eval_pred = get_loclin_pred(data.x_eval, model)
        y_mid_fit = LocLin(data.x, data.y_middle).vec_fit(data.x_eval)
        y_true_fit = LocLin(data.x, data.y).vec_fit(data.x_eval)
        yl_fit, yu_fit = None, None

    else:
        if method == "linear":
            model = LinearRegression()
        if method == "rf":
            model = RandomForestRegressor(
                n_estimators=rf_n_estimators, max_depth=rf_max_depth
            )
        if method == "krr":
            model = KernelRidge(alpha=krr_alpha, kernel=krr_kernel)

        model.fit(data.x_train, data.yd_train)
        y_test_pred = model.predict(data.x_test).T
        y_eval_pred = model.predict(data.x_eval).T

        y_mid_fit = model.fit(data.x, data.y_middle).predict(data.x_eval)
        y_true_fit = model.fit(data.x, data.y).predict(data.x_eval)
        yl_fit = model.fit(data.x, data.yl).predict(data.x_eval)
        yu_fit = model.fit(data.x, data.yu).predict(data.x_eval)
    return y_test_pred, y_eval_pred, y_mid_fit, y_true_fit, yl_fit, yu_fit


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


def compute_score(y_test_pred, yu_test, yl_test):
    diff_l = np.maximum(yl_test - y_test_pred, 0) ** 2
    diff_u = np.maximum(y_test_pred - yu_test, 0) ** 2
    return (diff_l + diff_u).mean(axis=1)


def get_loclin_pred(eval, loclin_models):
    pred = np.column_stack([mm.vec_fit(eval) for mm in loclin_models]).T
    return pred
