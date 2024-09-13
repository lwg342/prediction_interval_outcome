import os
import pandas as pd
import numpy as np
from utils.empirical import analyze_and_plot
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

current_date = pd.Timestamp.now().strftime("%Y-%m-%d")

data = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")
df = data[~data["Is_holdout"]].copy()
df_holdout = data[data["Is_holdout"]].copy()

result_file_path = f"asec23pub_results_{current_date}.csv"
bandwidth = np.array([0.34287434, 2.00268184])

def execution(current_date, df, result_file_path, bandwidth, alpha, exp_fixed, random_seed):
    np.random.seed(random_seed)
    _, results = analyze_and_plot(
        df,
        alpha=alpha,
        exp_fixed=exp_fixed,
        bandwidth=bandwidth,
        conformal_method="local",
        comment="full",
    )

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

# Function to run a single configuration in parallel
def run_parallel(iteration):
    for alpha in np.array([0.1, 0.5, 0.9]):
        for exp_fixed in np.array([10, 20]):
            random_seed = np.array(iteration) + 43
            execution(current_date, df, result_file_path, bandwidth, alpha, exp_fixed, random_seed)

# Use ProcessPoolExecutor for parallel processing
if __name__ == "__main__":
    iterations = range(20)  # 20 iterations

    # Set the number of CPU cores to use
    num_cores = 4  # Adjust this to the number of cores you want to use

    # Run in parallel across random seeds
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(tqdm(executor.map(run_parallel, iterations), total=len(iterations)))