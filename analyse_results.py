# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.empirical import visualize_prediction

dataset = "LFS"

if dataset == "LFS":  # 🇬🇧 UKDA_9248 data
    results = pd.read_csv("UKDA_9248_results_2024-08-27.csv")
    data = pd.read_csv("wage-data/clean_apsp_jd23_eul_pwta22.tab")
if dataset == "CPS":  # 🇺🇸 US CPS data
    results = pd.read_csv("asec23pub_results_2024-08-26.csv")
    data = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")

results = results.drop_duplicates(
    subset=["Education", "Alpha", "Experience", "Random Seed"], keep="first"
)

df_holdout = data[data["Is_holdout"]]
numeric_columns = results.select_dtypes(include="number").columns.tolist()

# Group by the relevant columns and calculate the mean of numeric columns
results_mean = results.groupby(["Education", "Alpha", "Experience"], as_index=False)[
    numeric_columns
].mean()
results_mean = results.loc[results["Education"] != 11]
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
selected_data = results_[columns_to_include]

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
    results_[col] = results_[col].apply(lambda x: (np.exp(x)))
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
                "Alpha": f"{row['Alpha']:.1f}",
                "Experience": row["Experience"],
                "Education": row["Education"],
                "Bounds": f"({int(row[col_lower])}, {int(row[col_upper])})",
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
df = pd.DataFrame(rows)

df.Education = df["Education"].astype(int)
df.Experience = df["Experience"].astype(int)
df.Bounds = df["Bounds"].astype(str)
results_formatted = df.pivot(
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
    def is_exact(df, dataset):
        if dataset == "LFS":
            return df["BANDG"] < 0
        elif dataset == "CPS":
            return df["I_ERNVAL"] == 0
        return pd.Series([False] * len(df))

    def calculate_coverage(df, lower_bound, upper_bound):
        lower_cov = df["Lower_bound"] >= lower_bound
        upper_cov = df["Upper_bound"] <= upper_bound
        return sum(lower_cov & upper_cov) / len(df) if len(df) > 0 else 0

    selection = (
        (df_holdout["Education"] == row["Education"])
        & (df_holdout["Experience"] >= row["Experience"] - 5)
        & (df_holdout["Experience"] <= row["Experience"] + 5)
    )

    if_exact = is_exact(df_holdout, dataset)

    df_temp = df_holdout[selection]
    df_temp_only_exact = df_holdout[selection & if_exact]

    row["Coverage_conformal"] = calculate_coverage(
        df_temp,
        float(row["Conformal Prediction Lower Bound"]),
        float(row["Conformal Prediction Upper Bound"]),
    )
    row["Coverage_conformal_with_exact_number"] = calculate_coverage(
        df_temp,
        float(row["Conformal Prediction Lower Bound with Exact Number"]),
        float(row["Conformal Prediction Upper Bound with Exact Number"]),
    )
    row["Coverage_conformal_of_exact"] = calculate_coverage(
        df_temp_only_exact,
        float(row["Conformal Prediction Lower Bound"]),
        float(row["Conformal Prediction Upper Bound"]),
    )
    row["Coverage_conformal_with_exact_number_of_exact"] = calculate_coverage(
        df_temp_only_exact,
        float(row["Conformal Prediction Lower Bound with Exact Number"]),
        float(row["Conformal Prediction Upper Bound with Exact Number"]),
    )

    return row


# Applying the function to each row
result_with_cov = results.apply(compute_coverage, axis=1)
# %%
exp = 20.0
alpha = result_with_cov["Alpha"].unique()[0]

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
# plt.ylim(0.0, 1)
plt.xlabel("Education")
plt.ylabel("Coverage")
# plt.title("Coverage Metrics by Education")
plt.legend()
plt.savefig(f"coverage-{exp}-{alpha}.pdf")
plt.show()


# plt.figure()

# # Plot Coverage_conformal

# # Plot Coverage_conformal
# results_with_cov_plot.plot(
#     x="Education",
#     y="Coverage_conformal_of_exact",
#     kind="line",
#     marker="o",
#     label="Coverage_conformal",
#     ax=plt.gca(),  # Plot on the current axes
# )

# # Plot Coverage_conformal_with_exact_number
# results_with_cov_plot.plot(
#     x="Education",
#     y="Coverage_conformal_with_exact_number_of_exact",
#     kind="line",
#     marker="o",
#     label="Coverage_conformal_with_exact_number",
#     ax=plt.gca(),  # Plot on the same axes
# )
# plt.axhline(1 - alpha, color="tab:red", linestyle="--")
# plt.ylim(0.0, 1)
# plt.xlabel("Education")
# plt.ylabel("Coverage")
# plt.title("Coverage Metrics by Education")
# plt.legend()
# plt.show()


# %%
