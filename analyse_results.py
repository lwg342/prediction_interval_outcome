# %%
import pandas as pd
import numpy as np

data = pd.read_csv("UKDA_9248_results.csv")
data
# %%
columns_to_include = [
    "Alpha",
    "Education",
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
]
selected_data = data[columns_to_include]

for col in [
    "Prediction Lower Bound",
    "Prediction Upper Bound",
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
]:
    data[col] = data[col].apply(lambda x: np.exp(x))

# %%
import pandas as pd

# Create a list to store the new rows
rows = []


# Define a function to append rows to the list
def append_rows(df, col_lower, col_upper, label):
    for _, row in df.iterrows():
        rows.append(
            {
                "Alpha": row["Alpha"],
                "Education": row["Education"],
                "Bounds": f"({row[col_lower]}, {row[col_upper]})",
                "Method": label,
            }
        )


# Append rows for 'Prediction'
append_rows(data, "Prediction Lower Bound", "Prediction Upper Bound", "Prediction")

# Append rows for 'Conformal Prediction'
append_rows(
    data,
    "Conformal Prediction Lower Bound",
    "Conformal Prediction Upper Bound",
    "Conformal Prediction",
)

# Convert the list of rows to a new DataFrame
results_formatted = pd.DataFrame(rows).pivot(
    index=["Alpha", "Method"], columns="Education", values="Bounds"
)

# Print the new DataFrame
results_formatted
# %%
