# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.empirical import visualize_prediction, analyze_and_plot
from sklearn.model_selection import train_test_split

current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

data = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")
df = data[~data["Is_holdout"]].copy()
df_holdout = data[data["Is_holdout"]].copy()

result_file_path = f"asec23pub_results_{current_date}.csv"
current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

df["Log_upper_bound"].describe()
# %%
bandwidth = np.array([0.34287434, 2.00268184])
for random_seed in np.arange(10) + 43:
    for alpha in np.array([0.1, 0.5, 0.9]):
        for exp_fixed in np.array([10, 20]):
            print(f"alpha: {alpha}, exp_fixed: {exp_fixed}, for seed: {random_seed}")
            np.random.seed(random_seed)
            dt, results = analyze_and_plot(
                df,
                alpha=alpha,
                exp_fixed=exp_fixed,
                bandwidth=bandwidth,
                # [!] bandwidth = None for running cross validation, result in [0.34287434 2.00268184]
                conformal_method="local",
                comment="full",
            )

            # np.random.seed(random_seed)
            # dt_exact, results_exact = analyze_and_plot(
            #     df.loc[df["I_ERNVAL"] == 0],
            #     alpha=alpha,
            #     exp_fixed=exp_fixed,
            #     bandwidth=bandwidth,
            #     conformal_method="local",
            #     comment="exact",
            # )

            np.random.seed(random_seed)
            dt_mid, results_mid = analyze_and_plot(
                df,
                alpha=alpha,
                exp_fixed=exp_fixed,
                yl_col="Log_mid_point",
                yu_col="Log_mid_point",
                bandwidth=bandwidth,
                conformal_method="local",
                comment="mid",
            )
            print(pd.DataFrame(results))
            print(pd.DataFrame(results_mid))

            rslt_collection = pd.concat(
                [
                    pd.DataFrame(results),
                    # pd.DataFrame(results_exact),
                    pd.DataFrame(results_mid),
                ]
            )

            rslt_collection["Random Seed"] = random_seed
            rslt_collection["Date"] = current_date
            rslt_collection["Experience Bandwidth"] = bandwidth[1]

            # File path for saving CSV

            if not os.path.isfile(result_file_path):
                rslt_collection.to_csv(
                    result_file_path, mode="w", header=True, index=False
                )
            else:
                rslt_collection.to_csv(
                    result_file_path, mode="a", header=False, index=False
                )


# %%
