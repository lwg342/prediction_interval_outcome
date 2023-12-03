import numpy as np
import matplotlib.pyplot as plt


def compute_score(y_test_prediction, yu_test, yl_test):
    diff_l = np.maximum(yl_test - y_test_prediction, 0) ** 2
    diff_u = np.maximum(y_test_prediction - yu_test, 0) ** 2
    return (diff_l + diff_u).mean(axis=1)


def get_interval(y, interval_bias=[0.0, 0.0]):
    yl = np.floor(y) - interval_bias[0]
    yu = np.ceil(y) + interval_bias[1]
    return yl, yu


def gen_x(n, k, x_distri):
    if x_distri == "normal":
        x = np.random.normal(0, 1, (n, k))
    if x_distri == "uniform":
        x = np.random.uniform(-np.sqrt(3), np.sqrt(3), (n, k))
    return x


def gen_noise(n, var_epsilon, epsilon_distri, df):
    if epsilon_distri == "normal":
        epsilon = np.random.normal(0, var_epsilon, n)
    if epsilon_distri == "chisquare":
        epsilon = (np.random.chisquare(df, n) - df) / np.sqrt(2 * df)
    if epsilon_distri == "no_noise":
        epsilon = np.zeros(n)
    return epsilon


def gen_outcomes(get_interval, cal_y_signal, beta, interval_bias, x, epsilon):
    y = cal_y_signal(x, beta) + epsilon
    yl, yu = get_interval(y, interval_bias=interval_bias)
    y_middle = (yl + yu) / 2
    return y, yl, yu, y_middle


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
    n,
    M,
    k,
    interval_bias,
    var_epsilon,
    x_distri,
    epsilon_distri,
    df,
    indices,
    evaluation_points,
    y_signal,
    y_eval_pred,
    y_pred_middle=None,
    filename=None,
    **kwargs,
):
    plt.figure()

    plt.plot(evaluation_points[:, -1], y_signal, label="y_signal")
    yl_signal, yu_signal = get_interval(y_signal, interval_bias=[0.0, 0.0])
    plt.plot(evaluation_points[:, -1], yl_signal, label="y_interval", color="green")
    plt.plot(evaluation_points[:, -1], yu_signal, label="y_interval", color="green")

    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred.min(axis=0),
        color="red",
        label="pre-selection prediction",
    )
    plt.plot(evaluation_points[:, -1], y_eval_pred.max(axis=0), color="red")

    plt.plot(
        evaluation_points[:, -1],
        y_eval_pred[indices].min(axis=0),
        "+",
        color="orange",
        label="post-section prediction",
    )
    plt.plot(
        evaluation_points[:, -1], y_eval_pred[indices].max(axis=0), "+", color="orange"
    )
    plt.xlabel("f$x_{k}$")
    if y_pred_middle is not None:
        plt.plot(
            evaluation_points[:, -1],
            y_pred_middle,
            label="fit y_middle",
            color="purple",
        )

    plt.title(
        f"$n$={n}, $M$={M}, $k$={k}, $v_\epsilon$={var_epsilon}, $b_1$={interval_bias[0]}, $b_2$={interval_bias[1]}, \n $x$_distri={x_distri}, $\epsilon$_distri={epsilon_distri}, df={df}"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def create_param_dict(
    n, M, k, beta, var_epsilon, interval_bias, x_distri, epsilon_distri, df
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
    }
    return params_dict
