import pandas as pd
import matplotlib.pyplot as plt
# %%
# [-] Analyse the results
# rslt_df = pd.read_csv("simulation_results_0.1_20240922-000223_fixed_censoring.csv")
rslt_df = pd.read_csv(
    "simulation_results_0.1_20241202-154712_random_censoring_1_chisquare.csv"
)
# %%
df_avg = (
    rslt_df.groupby("x_eval")
    .agg(
        {
            "vol_cdf": "mean",
            "vol_cdf_local": "mean",
            "vol_quantile_3": "mean",
            "vol_quantile_2": "mean",
            "cov_cdf": "mean",
            "cov_cdf_local": "mean",
            "cov_quantile_3": "mean",
            "cov_quantile_2": "mean",
        }
    )
    .reset_index()
)

plt.figure()
df_avg.loc[(df_avg["x_eval"] > -1.45) & (df_avg["x_eval"] < 1.45)].plot(
    x="x_eval", y=["vol_cdf", "vol_cdf_local", "vol_quantile_3"]
)
plt.savefig(f"volume_simulation_{rslt_df['alpha'][0]}.pdf")
plt.show()
df_avg[["x_eval", "vol_cdf", "vol_cdf_local", "vol_quantile_3", "vol_quantile_2"]]
# %%
plt.figure()
ax = df_avg.loc[(df_avg["x_eval"] > -1.45) & (df_avg["x_eval"] < 1.45)].plot(
    x="x_eval",
    y=["cov_cdf", "cov_cdf_local", "cov_quantile_3", "cov_quantile_2"],
    label=["CP", "Local CP", "CQR Quadratic", "CQR Cubic"],
    linestyle="--",
)
ax.set_xlabel("X")
plt.hlines(1 - rslt_df["alpha"][0], -1.5, 1.5, color="black", linestyle="-")
plt.ylim([0.7, 1.0])
plt.savefig(f"coverage_simulation_{rslt_df['alpha'][0]}.pdf")
plt.show()
# %%
plt.figure()
vol_sum = rslt_df.groupby("iteration").agg(
    {
        "vol_cdf": "sum",
        "vol_cdf_local": "sum",
        "vol_quantile_2": "sum",
        "vol_quantile_3": "sum",
    }
)
vol_avg = vol_sum * 3 / 50
plt.figure()
plt.boxplot(
    vol_avg,
    labels=["CP", "Local CP", "CQR Quadratic", "CQR Cubic"],
)
plt.savefig(f"volume_simulation_boxplot_{rslt_df['alpha'][0]}.pdf")
plt.show()
# %%
