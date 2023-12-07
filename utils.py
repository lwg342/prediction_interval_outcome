import numpy as np
import matplotlib.pyplot as plt


def compute_score(y_test_prediction, yu_test, yl_test):
    diff_l = np.maximum(yl_test - y_test_prediction, 0) ** 2
    diff_u = np.maximum(y_test_prediction - yu_test, 0) ** 2
    return (diff_l + diff_u).mean(axis=1)


def get_interval(y, scale=1.0, interval_bias=[0.0, 0.0]):
    yl = np.floor(y / scale) * scale - interval_bias[0]
    yu = np.ceil(y / scale) * scale + interval_bias[1]
    return yl, yu


def gen_x(N, k, x_distri):
    if x_distri == "normal":
        x = np.random.normal(0, 1, (N, k))
    if x_distri == "uniform":
        x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (N, k))
    return x


def gen_noise(n, var_epsilon, epsilon_distri, df):
    if epsilon_distri == "normal":
        epsilon = np.random.normal(0, var_epsilon, n)
    if epsilon_distri == "chisquare":
        epsilon = (np.random.chisquare(df, n) - df) / np.sqrt(2 * df)
    if epsilon_distri == "no_noise":
        epsilon = np.zeros(n)
    return epsilon


def gen_outcomes(get_interval, cal_y_signal, beta, interval_bias, x, epsilon, scale=1):
    y = cal_y_signal(x, beta) + epsilon
    yl, yu = get_interval(y, scale=scale, interval_bias=interval_bias)
    y_middle = (yl + yu) / 2
    return y, yl, yu, y_middle


def calculate_weights_and_yd(M, N, yl, yu, beta_a=1, beta_b=1):
    weights = np.random.beta(beta_a, beta_b, [M, N])
    yd = weights * yl + (1 - weights) * yu
    return weights, yd


def gen_eval(cal_y_signal, k, beta, n_test_points, x):
    evaluation_points = np.column_stack(
        (
            np.ones((n_test_points, k - 1)) * 0,
            np.linspace(np.min(x[:, -1]), np.max(x[:, -1]), n_test_points),
        )
    )
    y_eval_signal = cal_y_signal(evaluation_points, beta)
    return evaluation_points, y_eval_signal


def get_loclin_pred(eval, loclin_models):
    pred = np.column_stack([mm.vec_fit(eval) for mm in loclin_models]).T
    return pred


def select_indices(compute_score, tolerance, yu_test, yl_test, y_test_pred):
    score = compute_score(y_test_pred, yu_test, yl_test)
    smallest_score = np.min(score)
    threshold = smallest_score + tolerance
    indices = np.where(score < threshold)[0]
    return indices


def plot_result(
    get_interval,
    N,
    M,
    K,
    interval_bias,
    var_epsilon,
    x_distri,
    epsilon_distri,
    df,
    scale,
    indices,
    evaluation_points,
    y_eval_signal,
    y_eval_pred,
    y_mid_fit=None,
    y_true_fit=None,
    yl_fit=None,
    yu_fit=None,
    filename=None,
    **kwargs,
):
    plt.figure()

    plt.plot(
        evaluation_points[:, -1], y_eval_signal, label="y_signal", color="tab:blue", linewidth=2
    )

    yl_signal, yu_signal = get_interval(
        y_eval_signal, interval_bias=[0.0, 0.0], scale=scale
    )
    plt.plot(
        evaluation_points[:, -1],
        yl_signal,
        color="tab:green",
        label="y_interval",
    )
    plt.plot(evaluation_points[:, -1], yu_signal, color="tab:green")

    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred.min(axis=0),
        color="tab:orange",
        label="pre-selection prediction",
    )
    plt.plot(evaluation_points[:, -1], y_eval_pred.max(axis=0), color="tab:orange")

    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred[indices].min(axis=0),
        "--",
        color="tab:red",
        label="post-section prediction",
    )
    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred[indices].max(axis=0),
        "--",
        color="tab:red",
    )

    plt.plot(
        evaluation_points[:, -1],
        y_true_fit,
        label="fit y_true",
        color="tab:cyan",
    )

    if yl_fit is not None:
        plt.plot(evaluation_points[:, -1], yl_fit, label="fit yl", color="#E7D2CC")
    if yu_fit is not None:
        plt.plot(evaluation_points[:, -1], yu_fit, label="fit yu", color="#E7D2CC")

    if y_mid_fit is not None:
        plt.plot(
            evaluation_points[:, -1], y_mid_fit, label="fit y_middle", linestyle="--", color ="#BEB8DC"
        )

    plt.xlabel("f$x_{k}$")
    plt.title(
        f"$N$={N}, $M$={M}, $K$={K}, $v_\epsilon$={var_epsilon}, $b_1$={interval_bias[0]}, $b_2$={interval_bias[1]}, \n $x$_distri={x_distri}, $\epsilon$_distri={epsilon_distri}, df={df}, scale = {scale}"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def create_param_dict(
    n,
    M,
    k,
    beta,
    var_epsilon,
    interval_bias,
    x_distri,
    epsilon_distri,
    df,
    tolerance,
    n_test_points,
    evaluation_points,
    y_eval_signal,
):
    params_dict = {
        "n": n,
        "M": M,
        "k": k,
        "beta": beta,
        "var_epsilon": var_epsilon,
        "interval_bias": interval_bias,
        "x_distri": x_distri,
        "epsilon_distri": epsilon_distri,
        "df": df,
        "tolerance": tolerance,
        "n_test_points": n_test_points,
        "evaluation_points": evaluation_points,
        "y_eval_signal": y_eval_signal,
    }
    return params_dict
