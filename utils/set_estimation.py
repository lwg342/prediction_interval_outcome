import numpy as np


def find_optimal_interval(
    grid_of_intervals: np.ndarray,
    weights: np.ndarray,
    yl: float,
    yu: float,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Find the optimal interval from a grid of intervals based on weights and constraints.
    Parameters:
    - grid_of_intervals (numpy.ndarray): A grid of intervals represented as a 2D numpy array.
    - weights (numpy.ndarray): An array of weights.
    - yl (float): Lower bound constraint.
    - yu (float): Upper bound constraint.
    - alpha (float, optional): The significance level. Defaults to 0.1.
    Returns:
    - numpy.ndarray: The optimal interval that has the smallest width and satisfies the weight sum constraint.
    """
    interval_width = grid_of_intervals[:, 1] - grid_of_intervals[:, 0]
    # sort grid_of_intervals  by interval_width in ascending order
    grid_of_intervals = grid_of_intervals[np.argsort(interval_width)]
    for interval in grid_of_intervals:
        if weights_sum(weights, interval, yl, yu) >= 1 - alpha:
            return interval


def find_optimal_set(
    interval_grid: np.ndarray,
    weights: np.ndarray,
    yl: float,
    yu: float,
    alpha: float = 0.9,
    K: int = 1,
    margin: float = None,
) -> list[np.ndarray]:
    """
    Selects the optimal set of intervals from a grid of intervals based on certain criteria.
    Parameters:
    - interval_grid (numpy.ndarray): Grid of intervals.
    - weights (numpy.ndarray): Array of weights.
    - yl (float): Lower bound.
    - yu (float): Upper bound.
    - alpha (float): Threshold value.
    - K (int): Number of intervals to select.
    - margin (float): Margin value.
    Returns:
    - optim_set (list): List of selected intervals.
    """
    interval_width = interval_grid[:, 1] - interval_grid[:, 0]
    if margin is None:
        margin = np.min(interval_width) * 2
    # sort grid_of_intervals  by interval_width in ascending order

    interval_grid_sorted = interval_grid[np.argsort(interval_width)]
    np.sort(interval_width)
    interval_width = interval_grid_sorted[:, 1] - interval_grid_sorted[:, 0]

    min_width = np.inf
    for intv in interval_grid_sorted:
        if weights_sum(weights, intv, yl, yu) >= 1 - alpha:
            optim_interval = intv
            min_width = cal_width(optim_interval)
            break
    if K == 1:
        return [optim_interval]

    if K == 2:
        # Each one of the two intervals must have a width < min_width
        interval_grid_2 = interval_grid_sorted[
            (interval_width < min_width) & (interval_width > margin)
        ]

        # The sum of the weights of the two intervals must be >= 1-alpha
        weights_2 = np.array(
            [weights_sum(weights, interval, yl, yu) for interval in interval_grid_2]
        )
        width_2 = interval_grid_2[:, 1] - interval_grid_2[:, 0]

        cond_enough_weights = weights_2 > 1 - alpha - weights_2.max()
        cond_small_width = width_2 < min_width / 2

        interval_grid_2_enough_weights = interval_grid_2[cond_enough_weights]
        weights_2_enough_weights = weights_2[cond_enough_weights]
        width_2_enough_weights = width_2[cond_enough_weights]

        interval_pair_small = interval_grid_2[cond_small_width & cond_enough_weights]
        weights_2_small = weights_2[cond_small_width & cond_enough_weights]

        width_2_small = interval_pair_small[:, 1] - interval_pair_small[:, 0]

        optim_set = np.concatenate(
            [optim_interval[[0]], np.array([np.nan, np.nan]), optim_interval[[1]]]
        )
        width_thresh = min_width - margin

        for i, intv1 in enumerate(interval_pair_small):
            interval_width_i = width_2_small[i]
            for j, intv2 in enumerate(interval_grid_2_enough_weights):
                interval_width_j = width_2_enough_weights[j]
                if interval_width_i + interval_width_j < width_thresh:
                    if weights_2_small[i] + weights_2_enough_weights[j] >= 1 - alpha:
                        if not is_overlap(intv2, intv1):
                            width_thresh = interval_width_i + interval_width_j
                            optim_set = np.sort(np.concatenate([intv1, intv2]))
                            break
                else:
                    break
        return optim_set


def is_subinterval(N_intervals: np.ndarray, interval2: tuple) -> np.ndarray:
    """
    Determines which rows of a N*2 array of intervals are a subset of a given interval2.
    Parameters:
    N_intervals (numpy.ndarray): N*2 array of intervals.
    interval2 (tuple): Interval to compare against.
    Returns:
    numpy.ndarray: Boolean array indicating which rows of N_intervals are a subset of interval2.
    """
    return np.logical_and(
        N_intervals[:, 0] >= interval2[0], N_intervals[:, 1] <= interval2[1]
    )


def is_subset_of_any(N_intervals: np.ndarray, list_intervals: list) -> bool:
    """
    Check if each row of a N*2 array of intervals is a subset of any of the intervals in a list of intervals.
    Parameters:
    N_intervals (numpy.ndarray): N*2 array of intervals.
    list_intervals (list): List of intervals.
    Returns:
    bool: True if any row of N_intervals is a subset of any interval in list_intervals, False otherwise.
    """
    return np.any(
        [is_subinterval(N_intervals, interval) for interval in list_intervals], axis=0
    )


def weights_sum(weights, interval, yl, yu):
    mask = (interval[0] <= yl) & (yu <= interval[1])
    return np.sum(weights[mask])


def eligible_t0t1(yl, yu, weights):
    positive_weights_indices = weights > 0
    t0 = np.sort(yl[positive_weights_indices])
    t1 = np.sort(yu[positive_weights_indices])[::-1]
    return t0, t1


def create_grid(t0, t1, n_grid=100):
    return np.linspace(t0.min(), t1.max(), n_grid)


def create_valid_interval_array(t0, t1):
    # return a list of intervals with endpoints from t0 and t1, with t0 < t1
    return np.array([[t0_i, t1_i] for t0_i in t0 for t1_i in t1 if t0_i < t1_i])


is_overlap = (
    lambda interval1, interval2: interval1[0] <= interval2[1]
    and interval1[1] >= interval2[0]
)
cal_width = lambda interval: interval[1] - interval[0]
