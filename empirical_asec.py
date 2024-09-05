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
for random_seed in np.arange(20) + 43:
    for alpha in np.array([0.1, 0.5, 0.9]):
        for exp_fixed in np.array([10, 20]):
            print(f"alpha: {alpha}, exp_fixed: {exp_fixed}")
            np.random.seed(random_seed)
            dt, results = analyze_and_plot(
                df,
                alpha=alpha,
                exp_fixed=exp_fixed,
                bandwidth=bandwidth,
                # [!] bandwidth = None for running cross validation, result in [0.34287434 2.00268184]
                conformal_method="local",
            )

            np.random.seed(random_seed)
            dt_exact, results_exact = analyze_and_plot(
                df.loc[df["I_ERNVAL"] == 0],
                alpha=alpha,
                exp_fixed=exp_fixed,
                bandwidth=bandwidth,
                conformal_method="local",
            )
            # dt_imputed, results_imputed = analyze_and_plot(
            #     df,
            #     yl_col="Log_income_with_imputed_lower",
            #     yu_col="Log_income_with_imputed_upper",
            #     alpha=alpha,
            # )
            dt_mid, results_mid = analyze_and_plot(
                df,
                yl_col="Log_mid_point",
                yu_col="Log_mid_point",
                alpha=alpha,
            )
            results["random seed"] = random_seed
            df_results = pd.DataFrame(
                {
                    "Education": results["edu"],
                    "Experience": [results["experience"]] * len(results["edu"]),
                    "Alpha": [results["alpha"]] * len(results["edu"]),
                    "Prediction Lower Bound": results["prediction interval"][0],
                    "Prediction Upper Bound": results["prediction interval"][1],
                    "Conformal Prediction Lower Bound": results[
                        "conformal prediction interval"
                    ][0],
                    "Conformal Prediction Upper Bound": results[
                        "conformal prediction interval"
                    ][1],
                    "Conformal Correction": results["conformal correction"],
                    "Prediction Lower Bound with Exact Number": results_exact[
                        "prediction interval"
                    ][0],
                    "Prediction Upper Bound with Exact Number": results_exact[
                        "prediction interval"
                    ][1],
                    "Conformal Prediction Lower Bound with Exact Number": results_exact[
                        "conformal prediction interval"
                    ][0],
                    "Conformal Prediction Upper Bound with Exact Number": results_exact[
                        "conformal prediction interval"
                    ][1],
                    "Prediction Lower Bound with Mid Point": results_mid[
                        "prediction interval"
                    ][0],
                    "Prediction Upper Bound with Mid Point": results_mid[
                        "prediction interval"
                    ][1],
                    "Conformal Prediction Lower Bound with Mid Point": results_mid[
                        "conformal prediction interval"
                    ][0],
                    "Conformal Prediction Upper Bound with Mid Point": results_mid[
                        "conformal prediction interval"
                    ][1],
                    
                    "Conformal Correction": results_exact["conformal correction"],
                    "Kernel Regression Lower": results["kernel regression yl"],
                    "Kernel Regression Upper": results["kernel regression yu"],
                    "Random Seed": [results["random seed"]] * len(results["edu"]),
                    "Date": [current_date] * len(results["edu"]),
                    "Experience Bandwidth": [bandwidth[1]] * len(results["edu"]),
                }
            )

            # File path for the CSV

            if not os.path.isfile(result_file_path):
                df_results.to_csv(result_file_path, mode="w", header=True, index=False)
            else:
                df_results.to_csv(result_file_path, mode="a", header=False, index=False)


# %%
