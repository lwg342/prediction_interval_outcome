# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.empirical import visualize_prediction

dataset = "LFS"

if dataset == "LFS":  # 🇬🇧 UKDA_9248 data
    results = pd.read_csv("UKDA_9248_results_2024-08-23.csv")
    data = pd.read_csv("wage-data/clean_apsp_jd23_eul_pwta22.tab")
if dataset == "CPS":  # 🇺🇸 US CPS data
    results = pd.read_csv("asec23pub_results_20240819.csv")
    data = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")

results = results.drop_duplicates(
    subset=["Education", "Alpha", "Experience"], keep="first"
)

df_holdout = data[data["Is_holdout"]]

results
# %%
columns_to_include = [
    "Alpha",
    "Education",
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "Prediction Lower Bound with Exact Number",
    "Prediction Upper Bound with Exact Number",
    "Conformal Prediction Lower Bound with Exact Number",
    "Conformal Prediction Upper Bound with Exact Number",
]
selected_data = results[columns_to_include]

for col in [
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "Prediction Lower Bound with Exact Number",
    "Prediction Upper Bound with Exact Number",
    "Conformal Prediction Lower Bound with Exact Number",
    "Conformal Prediction Upper Bound with Exact Number",
]:
    results[col] = results[col].apply(lambda x: f"{round(np.exp(x))}")
    # data[col] = data[col].apply(lambda x: x)


# %%
# [-] Create table of results
import pandas as pd

# Create a list to store the new rows
rows = []


# Define a function to append rows to the list
def append_rows(df, col_lower, col_upper, label):
    for _, row in df.iterrows():
        rows.append(
            {
                "Alpha": f"{row['Alpha']:.2f}",
                "Experience": row["Experience"],
                "Education": row["Education"],
                "Bounds": f"({row[col_lower]}, {row[col_upper]})",
                "Method": label,
            }
        )


# Append rows for 'Prediction'
append_rows(results, "Prediction Lower Bound", "Prediction Upper Bound", "P")

# Append rows for 'Conformal Prediction'
append_rows(
    results,
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "CP",
)

append_rows(
    results,
    "Prediction Lower Bound with Exact Number",
    "Prediction Upper Bound with Exact Number",
    "PW",
)
append_rows(
    results,
    "Conformal Prediction Lower Bound with Exact Number",
    "Conformal Prediction Upper Bound with Exact Number",
    "CPW",
)
# Convert the list of rows to a new DataFrame
results_formatted = pd.DataFrame(rows).pivot(
    index=["Alpha", "Method", "Experience"], columns="Education", values="Bounds"
)

latex_table = results_formatted.to_latex(
    multicolumn=True,
    multirow=True,
    label=f"tab:{dataset}",
    longtable=True,
    caption="Conformal prediction and prediction intervals for annual income, by education level and the speficied alpha level. In the method column, P stands for prediction and CP stands for conformal prediction. W indicates prediction without interval censored data. ",
)

print(latex_table)
# %%
# [-] Analyse coverage on the holdout data


def compute_coverage(row):
    selection = (
        (df_holdout["Education"] == row["Education"])
        & (df_holdout["Experience"] >= row["Experience"] - 3)
        & (df_holdout["Experience"] <= row["Experience"] + 3)
    )
    if_exact = df_holdout["BANDG"] < 0

    df_temp = df_holdout[selection]
    df_temp_only_exact = df_holdout[selection & if_exact]

    # [-] coverage of both exact numbers and intervals
    lower_cov = df_temp["Lower_bound"] >= float(row["Conformal Prediction Lower Bound"])
    upper_cov = df_temp["Upper_bound"] <= float(row["Conformal Prediction Upper Bound"])

    lower_cov_with_exact = df_temp["Lower_bound"] >= float(
        row["Conformal Prediction Lower Bound with Exact Number"]
    )
    upper_cov_with_exact = df_temp["Upper_bound"] <= float(
        row["Conformal Prediction Upper Bound with Exact Number"]
    )

    # [-] coverage of exactly reported
    lower_cov_of_exact = df_temp_only_exact["Lower_bound"] >= float(
        row["Conformal Prediction Lower Bound"]
    )
    upper_cov_of_exact = df_temp_only_exact["Upper_bound"] <= float(
        row["Conformal Prediction Upper Bound"]
    )

    lower_cov_with_exact_of_exact = df_temp_only_exact["Lower_bound"] >= float(
        row["Conformal Prediction Lower Bound with Exact Number"]
    )
    upper_cov_with_exact_of_exact = df_temp_only_exact["Upper_bound"] <= float(
        row["Conformal Prediction Upper Bound with Exact Number"]
    )

    total_num = df_temp.shape[0]
    total_num_of_exact = df_temp_only_exact.shape[0]
    if total_num > 0:
        row["Coverage_conformal"] = sum(lower_cov & upper_cov) / total_num
        row["Coverage_conformal_with_exact_number"] = (
            sum(lower_cov_with_exact & upper_cov_with_exact) / total_num
        )

        row["Coverage_conformal_of_exact"] = (
            sum(lower_cov_of_exact & upper_cov_of_exact) / total_num_of_exact
        )
        row["Coverage_conformal_with_exact_number_of_exact"] = (
            sum(lower_cov_with_exact_of_exact & upper_cov_with_exact_of_exact)
            / total_num_of_exact
        )

    else:
        row["Coverage_conformal"] = 0
        row["Coverage_conformal_with_exact_number"] = 0

    return row


# Applying the function to each row
result_with_cov = results.apply(compute_coverage, axis=1)
# %%
exp = 10
alpha = 0.5

results_with_cov_plot = result_with_cov.loc[
    (result_with_cov["Experience"] == exp) & (result_with_cov["Alpha"] == alpha),
].sort_values(by="Education")
import matplotlib.pyplot as plt

plt.figure()

# Plot Coverage_conformal

# Plot Coverage_conformal
results_with_cov_plot.plot(
    x="Education",
    y="Coverage_conformal",
    kind="line",
    marker="o",
    label="Coverage_conformal",
    ax=plt.gca(),  # Plot on the current axes
)

# Plot Coverage_conformal_with_exact_number
results_with_cov_plot.plot(
    x="Education",
    y="Coverage_conformal_with_exact_number",
    kind="line",
    marker="o",
    label="Coverage_conformal_with_exact_number",
    ax=plt.gca(),  # Plot on the same axes
)
plt.axhline(1 - alpha, color="tab:red", linestyle="--")
# plt.ylim(0.5, 1)
plt.xlabel("Education")
plt.ylabel("Coverage")
plt.title("Coverage Metrics by Education")
plt.legend()
plt.show()

# %%
