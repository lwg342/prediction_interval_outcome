import numpy as np
from utils import *
import tqdm


def cv_bandwidth(data, candidate_bandwidth, nfolds=5, alpha=0.05):
    # Equally split the training data into nfolds
    x_folds = np.array_split(data.x_train.to_numpy(), nfolds)
    yl_folds = np.array_split(data.yl_train.to_numpy(), nfolds)
    yu_folds = np.array_split(data.yu_train.to_numpy(), nfolds)
    coverage_results = np.full((candidate_bandwidth.shape[0], nfolds), np.nan)
    width_results = np.full((candidate_bandwidth.shape[0], nfolds), np.nan)
    for k, h in tqdm.tqdm(enumerate(candidate_bandwidth)):
        print(h, end="\r")
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
                x_val_cv, x_train_cv, yl_train_cv, yu_train_cv, h, alpha=alpha
            )

            # Compute the empirical coverage probability of the pred_interval
            coverage = np.mean(
                (pred_interval_at_fold[0] <= yl_val_cv)
                & (yu_val_cv <= pred_interval_at_fold[1])
            )

            # Collect the coverage in a list for h, and i
            coverage_results[k, i] = coverage
            width_results[k, i] = np.mean(
                pred_interval_at_fold[1] - pred_interval_at_fold[0]
            )
    # Compute the average coverage probability for each h and find the one that is closest to 1-alpha

    mse = np.mean((coverage_results - (1 - alpha)) ** 2, axis=1)
    mae = np.mean(np.abs(coverage_results - (1 - alpha)), axis=1)
    width_score = np.mean(width_results, axis=1)
    # print(f"Mean squared error: {mse}")
    # print(f"Mean width: {width_score}")

    best_h_width = candidate_bandwidth[np.argmin(width_score)]
    best_h_mse = candidate_bandwidth[np.argmin(mse)]
    best_h_mae = candidate_bandwidth[np.argmin(mae)]

    # find the best bandwidth that has smallest rank in mse and width_score
    rank_mse = np.argsort(np.argsort(mse))
    rank_width = np.argsort(np.argsort(width_score))
    rank = rank_mse + rank_width
    print(rank)

    print(
        f"Best bandwidth by width: {best_h_width}\nBest bandwidth by mse: {best_h_mse}\nBest bandwidth by mae: {best_h_mae}\nBest bandwidth by rank: {candidate_bandwidth[np.argmin(rank)]}"
    )
    return best_h_mse, coverage_results, width_score


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

    data = SimData(get_interval=get_interval)
    data.x_train.shape

    # h = silvermans_rule(data.x_train)
    candidate_bandwidth = 0.2 * np.arange(1, 20) * silvermans_rule(data.x_train)
    alpha = 0.05
    best_h, coverage_results, width_score = cv_bandwidth(
        data, candidate_bandwidth, nfolds=5, alpha=alpha
    )
    print(f"Best bandwidth: {best_h}")
