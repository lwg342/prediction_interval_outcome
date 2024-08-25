# %%
import numpy as np


score_func_abs_val = lambda y_obs, y_pred: np.abs(y_obs - y_pred)
score_func_sq = lambda y_obs, y_pred: (y_obs - y_pred) ** 2


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
    x_expanded = x[:, np.newaxis, :]  # Shape: (n, 1, k)
    xi_expanded = xi[np.newaxis, :, :]  # Shape: (1, n2, k)

    # Scale distances
    u = (x_expanded - xi_expanded) / h
    # Compute the product of the Epanechnikov kernel for each dimension
    kernel_values = epanechnikov_kernel(u)
    return np.prod(kernel_values, axis=2)


def compute_weights(weights_unnormalized):
    weights = weights_unnormalized / np.sum(weights_unnormalized, axis=1, keepdims=True)
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
    weights_pos = weights > 0
    yl_pos = yl_train[weights_pos]
    yu_pos = yu_train[weights_pos]
    weights_pos = weights[weights_pos]

    interval_width = np.inf
    optimal_t0, optimal_t1 = None, None
    for t0_i in t0c:
        for t1_i in t1c:
            if t0_i < t1_i:
                weight_sum = compute_weight_sum(
                    yl_pos,
                    yu_pos,
                    weights_pos,
                    t0_i,
                    t1_i,
                )
                if weight_sum >= 1 - alpha and t1_i - t0_i < interval_width:
                    interval_width = t1_i - t0_i
                    optimal_t0, optimal_t1 = t0_i, t1_i
                if weight_sum < 1 - alpha:
                    break
    return [optimal_t0, optimal_t1]


def pred_interval(x, x_train, yl_train, yu_train, h, alpha=0.05):
    pred_interval = np.zeros([2, x.shape[0]])
    weights = compute_weights(
        multivariate_epanechnikov_kernel(
            x,
            x_train,
            h=h,
        )
    )
    for i in range(x.shape[0]):
        t0c, t1c = eligible_t0t1(yl_train, yu_train, weights[i])

        pred_interval[:, i] = find_t_hat(
            yl_train, yu_train, compute_weight_sum, weights[i], t0c, t1c, alpha=alpha
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
