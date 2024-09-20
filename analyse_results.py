# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
dataset = "CPS"

if dataset == "LFS":  # 🇬🇧 UKDA_9248 data
    rslt0 = pd.read_csv("UKDA_9248_results_2024-08-28.csv")
    data = pd.read_csv("wage-data/clean_apsp_jd23_eul_pwta22.tab")
if dataset == "CPS":  # 🇺🇸 US CPS data
    rslt0 = pd.read_csv("asec23pub_results_2024-09-18.csv")
    data = pd.read_csv("wage-data/clean_data_asec_pppub23.csv")

rslt0 = rslt0.drop_duplicates(
    subset=["Education", "Alpha", "Experience", "Comment", "Random Seed"], keep="first"
)

numeric_columns = rslt0.select_dtypes(include="number").columns.tolist()

rslt0
results = rslt0.copy()
# %%
# [-] Turn income columns from log to normal scale
income_col = [
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    # "Kernel Regression Lower",
    # "Kernel Regression Upper",
]


results[income_col] = rslt0[income_col].apply(lambda x: (np.exp(x)).round())
results
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

    _, df_holdout = train_test_split(
        data, test_size=0.2, random_state=row["Random Seed"]
    )

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
    # row["Coverage_conformal_with_exact_number"] = calculate_coverage(
    #     df_temp,
    #     float(row["Conformal Prediction Lower Bound with Exact Number"]),
    #     float(row["Conformal Prediction Upper Bound with Exact Number"]),
    # )
    row["Coverage_conformal_of_exact"] = calculate_coverage(
        df_temp_only_exact,
        float(row["Conformal Prediction Lower Bound"]),
        float(row["Conformal Prediction Upper Bound"]),
    )
    # row["Coverage_conformal_with_exact_number_of_exact"] = calculate_coverage(
    #     df_temp_only_exact,
    #     float(row["Conformal Prediction Lower Bound with Exact Number"]),
    #     float(row["Conformal Prediction Upper Bound with Exact Number"]),
    # )

    return row


results = results.apply(compute_coverage, axis=1)

# %%
# [-] Create table of results

# [-] Group by the relevant columns and calculate the mean of numeric columns
numeric_columns = results.select_dtypes(include="number").columns.tolist()
results_mean = results.groupby(
    ["Education", "Alpha", "Experience", "Comment"], as_index=False
)[numeric_columns].mean()
results_mean = results_mean.loc[results_mean["Education"] != 11.0]

# %%
# [-] Create a list to store the new rows

rows = []


# Define a function to append rows to the list
def append_rows(df, col_lower, col_upper, label):
    for _, row in df.iterrows():
        rows.append(
            {
                "Alpha": f"{row['Alpha']:.1f}",
                "Experience": row["Experience"],
                "Education": row["Education"],
                "Bounds": f"({(row[col_lower]):,.0f} - {(row[col_upper]):,.0f})",
                "Method": label,
            }
        )


append_rows(
    results_mean.loc[results_mean["Comment"] == "full"],
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "P",
)

# Append rows for 'Conformal Prediction'
append_rows(
    results_mean.loc[results_mean["Comment"] == "full"],
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "CP",
)

append_rows(
    results_mean.loc[results_mean["Comment"] == "hot deck"],
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "PM",
)
append_rows(
    results_mean.loc[results_mean["Comment"] == "hot deck"],
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "CPM",
)
# Convert the list of rows to a new DataFrame
result_table = pd.DataFrame(rows)

result_table.Education = result_table["Education"].astype(int)
result_table.Experience = result_table["Experience"].astype(int)
result_table.Bounds = result_table["Bounds"].astype(str)
# %%


results_formatted = result_table.pivot(
    index=["Alpha", "Method", "Experience"], columns="Education", values="Bounds"
)

latex_table = results_formatted.to_latex(
    multicolumn=True,
    multirow=True,
    label=f"tab:{dataset}",
    longtable=True,
    # caption="Conformal prediction and prediction intervals for annual income, by education level and the speficied alpha level. In the method column, P stands for prediction and CP stands for conformal prediction. W indicates prediction without interval censored data. ",
    caption="Prediction intervals for annual income",
)

print(latex_table)
# %%
# %%

# [-] Analyse the coverage property

results_mean
results_cov = results_mean.loc[results_mean["Alpha"] == 0.9]
results_cov
# %%
exp = 40.0
alpha = results_mean["Alpha"].unique()[0]

results_with_cov_plot = results_mean.loc[
    (results_mean["Experience"] == exp) & (results_mean["Alpha"] == alpha),
].sort_values(by="Education")
using_impute = results_with_cov_plot["Comment"] == "hot deck"

import matplotlib.pyplot as plt

plt.figure()

# Plot Coverage_conformal

# Plot Coverage_conformal
results_with_cov_plot.loc[~using_impute].plot(
    x="Education",
    y="Coverage_conformal",
    kind="line",
    marker="o",
    label="Coverage_conformal",
    ax=plt.gca(),  # Plot on the current axes
)
# Plot Coverage_conformal_with_exact_number
results_with_cov_plot.loc[using_impute].plot(
    x="Education",
    y="Coverage_conformal",
    kind="line",
    marker="o",
    label="Coverage_conformal",
    ax=plt.gca(),  # Plot on the same axes
)
plt.axhline(1 - alpha, color="tab:red", linestyle="--")
plt.ylim(0.5, 1)
plt.xlabel("Education")
plt.ylabel("Coverage")
# plt.title("Coverage Metrics by Education")
plt.legend()
plt.savefig(f"coverage-{int(exp)}-{alpha:.2f}.pdf")
plt.show()

# %%
# Pivot table for Coverage_conformal
df = results_mean.copy()
df["Education"] = df["Education"].astype(int)
df["Experience"] = df["Experience"].astype(int)
df["Alpha"] = df["Alpha"].apply(lambda x: f"{x:.2f}")

df_pivot = df.pivot_table(
    index=[
        "Education",
        "Experience",
    ],
    columns=["Alpha", "Comment"],
    values="Coverage_conformal",
)

df_pivot = df_pivot.map(lambda x: f"{x:.3f}" if not pd.isnull(x) else "")


cov_latex = df_pivot.to_latex(multicolumn=True, multirow=True)
print(cov_latex)
# %% 


import numpy as np

# Create a copy of the pivot table
df_bold = df_pivot.copy()

# Iterate through each Alpha and Comment pair
for alpha in df["Alpha"].unique():
    alpha_val = float(alpha)  # Convert string alpha to float
    one_minus_alpha = 1 - alpha_val  # Calculate 1 - alpha

    # Iterate through each Education and Experience index
    for idx in df_pivot.index:
        full_val = df_pivot.loc[idx, (alpha, "full")]
        hot_deck_val = df_pivot.loc[idx, (alpha, "hot deck")]
        
        # Ensure the values are numeric
        if not pd.isnull(full_val) and not pd.isnull(hot_deck_val):
            full_diff = abs(one_minus_alpha - float(full_val))
            hot_deck_diff = abs(one_minus_alpha - float(hot_deck_val))

            # Bold the value closer to 1 - alpha
            if full_diff < hot_deck_diff:
                df_bold.loc[idx, (alpha, "full")] = f"\\textbf{{{full_val}}}"
            else:
                df_bold.loc[idx, (alpha, "hot deck")] = f"\\textbf{{{hot_deck_val}}}"

# Convert to LaTeX with bolded values
cov_latex_bold = df_bold.to_latex(multicolumn=True, multirow=True, escape=False, caption="Coverage of conformal prediction intervals")
print(cov_latex_bold)

# %%
