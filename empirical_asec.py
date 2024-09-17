# %%
import os
import pandas as pd
import numpy as np
from utils.empirical import analyze_and_plot

current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

data = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")
df = data[~data["Is_holdout"]].copy()
df_holdout = data[data["Is_holdout"]].copy()

result_file_path = f"asec23pub_results_{current_date}.csv"
current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

df["Log_upper_bound"].describe()
# %%
bandwidth = np.array([0.34287434, 2.00268184])
from tqdm import tqdm


def execution(
    current_date, df, result_file_path, bandwidth, alpha, exp_fixed, random_seed
):
    np.random.seed(random_seed)
    _, results = analyze_and_plot(
        df,
        alpha=alpha,
        exp_fixed=exp_fixed,
        bandwidth=bandwidth,
        # [!] bandwidth = None for running cross validation, result in [0.34287434 2.00268184]
        conformal_method="local",
        comment="full",
    )

    # np.random.seed(random_seed)
    # _, results_exact = analyze_and_plot(
    #     df.loc[df["I_ERNVAL"] == 0],
    #     alpha=alpha,
    #     exp_fixed=exp_fixed,
    #     bandwidth=bandwidth,
    #     conformal_method="local",
    #     comment="exact",
    # )

    # np.random.seed(random_seed)
    # _, results_mid = analyze_and_plot(
    #     df,
    #     alpha=alpha,
    #     exp_fixed=exp_fixed,
    #     yl_col="Log_mid_point",
    #     yu_col="Log_mid_point",
    #     bandwidth=bandwidth,
    #     conformal_method="local",
    #     comment="mid",
    # )

    np.random.seed(random_seed)
    _, results_impute = analyze_and_plot(
        df,
        alpha=alpha,
        exp_fixed=exp_fixed,
        yl_col="Log_income_with_imputed_lower",
        yu_col="Log_income_with_imputed_upper",
        bandwidth=bandwidth,
        conformal_method="local",
        comment="hot deck",
    )

    rslt_collection = pd.concat(
        [
            pd.DataFrame(results),
            # pd.DataFrame(results_exact),
            # pd.DataFrame(results_mid),
            pd.DataFrame(results_impute),
        ]
    )
    rslt_collection["Random Seed"] = random_seed
    rslt_collection["Date"] = current_date
    rslt_collection["Experience Bandwidth"] = bandwidth[1]

    # File path for saving CSV

    if not os.path.isfile(result_file_path):
        rslt_collection.to_csv(result_file_path, mode="w", header=True, index=False)
    else:
        rslt_collection.to_csv(result_file_path, mode="a", header=False, index=False)


for iter in tqdm(range(20)):
    for alpha in np.array([0.1, 0.5, 0.9]):
        for exp_fixed in np.array([10, 20]):
            random_seed = np.array(iter) + 19260817
            # print(f"alpha: {alpha}, exp_fixed: {exp_fixed}, for seed: {random_seed}")

            execution(
                current_date,
                df,
                result_file_path,
                bandwidth,
                alpha,
                exp_fixed,
                random_seed,
            )


# %%
