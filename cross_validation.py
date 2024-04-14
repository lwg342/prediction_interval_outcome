import numpy as np
from utils_sim import *


def cv_bandwidth(data, candidate_bandwidth, nfolds=5, alpha=0.05):
    # Equally split the training data into nfolds
    x_folds = np.array_split(data.x_train, nfolds)
    yl_folds = np.array_split(data.yl_train, nfolds)
    yu_folds = np.array_split(data.yu_train, nfolds)
    coverage_results = np.full((candidate_bandwidth.shape[0], nfolds), np.nan)
    for k, h in enumerate(candidate_bandwidth):
        # print(h)
        for i in range(nfolds):
            # Get the training data for the current fold
            x_train_cv = np.concatenate([x for j, x in enumerate(x_folds) if j != i])
            x_val_cv = x_folds[i]
            yl_train_cv = np.concatenate(
                [yl for j, yl in enumerate(yl_folds) if j != i]
            )
            yl_val_cv = yl_folds[i]
            yu_train_cv = np.concatenate(
                [yu for j, yu in enumerate(yu_folds) if j != i]
            )
            yu_val_cv = yu_folds[i]

            # Find the optimal t0 and t1 for the current fold
            pred_interval_at_fold = pred_interval(
                x_val_cv, x_train_cv, yl_train_cv, yu_train_cv, h
            )

            # Compute the empirical coverage probability of the pred_interval
            coverage = np.mean(
                (pred_interval_at_fold[0] <= yl_val_cv)
                & (yu_val_cv <= pred_interval_at_fold[1])
            )

            # Collect the coverage in a list for h, and i
            coverage_results[k, i] = coverage

    # Compute the average coverage probability for each h and find the one that is closest to 1-alpha

    mse = np.mean((coverage_results - (1 - alpha)) ** 2, axis=1)
    # best_h = candidate_bandwidth[np.argmax(avg_coverage)]
    best_h = candidate_bandwidth[np.argmin(mse)]
    return best_h, coverage_results


if __name__ == "__main__":

    def get_interval(y, scale, **kwargs):
        # err1 = np.random.chisquare(df=1, size=y.shape) * scale
        # err2 = np.random.chisquare(df=2, size=y.shape) * scale

        # err1 = np.random.exponential(0.1, size=y.shape)
        # err2 = np.random.exponential(0.5, size=y.shape)

        err1 = np.random.exponential(0.5, size=y.shape)
        err2 = np.random.exponential(1.0, size=y.shape)

        # err1 = 0.0
        # err2 = 0.0

        # err1 = np.random.uniform(0, 0.1, size=y.shape)
        # err2 = np.random.uniform(0, 4, size=y.shape)

        # err1 = np.abs(np.random.normal(0, scale * 0.5, size=y.shape))
        # err2 = np.abs(np.random.normal(0, scale * 10, size=y.shape))

        return (y - err1, y + err2)

    data = Data(get_interval=get_interval)
    data.x_train.shape

    # h = silvermans_rule(data.x_train)
    candidate_bandwidth = np.linspace(0.5, 5.0, 10) * silvermans_rule(data.x_train)

    best_h, avg_coverage = cv_bandwidth(data, candidate_bandwidth, nfolds=5)
