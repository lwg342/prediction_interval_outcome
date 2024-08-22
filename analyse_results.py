# %%
import pandas as pd
import numpy as np

# filename= "UKDA_9248_results_2024-08-14"
filename = "asec23pub_results_20240819"
data = pd.read_csv(f"{filename}.csv").drop_duplicates(
    subset=["Education", "Alpha", "Experience"], keep="first"
)
data
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
selected_data = data[columns_to_include]

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
    data[col] = data[col].apply(lambda x: f"{round(np.exp(x))}")
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
append_rows(data, "Prediction Lower Bound", "Prediction Upper Bound", "P")

# Append rows for 'Conformal Prediction'
append_rows(
    data,
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "CP",
)

append_rows(
    data,
    "Prediction Lower Bound with Exact Number",
    "Prediction Upper Bound with Exact Number",
    "PW",
)
append_rows(
    data,
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
