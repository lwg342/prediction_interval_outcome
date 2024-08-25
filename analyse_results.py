# %%
import pandas as pd
import numpy as np

# filename= "UKDA_9248_results_2024-08-14"
filename = "asec23pub_results_20240819"
results = pd.read_csv(f"{filename}.csv").drop_duplicates(
    subset=["Education", "Alpha", "Experience"], keep="first"
)
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
    label=f"tab:{filename}",
    longtable=True,
    caption="Conformal prediction and prediction intervals for annual income, by education level and the speficied alpha level. In the method column, P stands for prediction and CP stands for conformal prediction. W indicates prediction without interval censored data. ",
)

# %%
print(latex_table)
# %%
# [-] Plots
import pandas as pd

transform = np.array


plt.figure()

visualize_prediction(
    results,
    "prediction interval",
    transform,
    f"Prediction (exact and range)",
    "tab:blue",
    offset=-0.3,
)
visualize_prediction(
    results,
    "conformal prediction interval",
    transform,
    f"Conf. prediction (exact and range)",
    "tab:red",
    offset=-0.1,
)
print(
    np.exp(results["conformal prediction interval"][0]),
    np.exp(results["conformal prediction interval"][1]),
)

visualize_prediction(
    results_exact,
    "prediction interval",
    transform,
    f"Prediction (exact only)",
    "tab:orange",
    marker="s",
    offset=0.1,
)
visualize_prediction(
    results_exact,
    "conformal prediction interval",
    transform,
    f"Conf. prediction (exact only)",
    "tab:green",
    offset=0.3,
    marker="s",
)
print(
    np.exp(results_exact["conformal prediction interval"][0]),
    np.exp(results_exact["conformal prediction interval"][1]),
)
plt.xlabel("Education")
plt.ylabel("Predicted log earnings")


plt.plot(
    results["edu"],
    results["kernel regression yl"],
    label="Mean of lower bound",
    marker="x",
)
plt.plot(
    results["edu"],
    results["kernel regression yu"],
    label="Mean of upper bound",
    marker="x",
)
# plt.legend(loc="center right", bbox_to_anchor=(-0.5, -0.1), ncol=1)
# Place the legend outside the plot area
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.title(
    f"Conformal Prediction Intervals when Experience is fixed at {results['experience']}"
)
cdt = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"empirical_edu_lfs_{alpha}_{cdt}.pdf", bbox_inches="tight")
