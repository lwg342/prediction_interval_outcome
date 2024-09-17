# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Parameters for the first skewed normal distribution
a1, loc1, scale1 = 5, -1.5, 1  # positive a for right skew
# Parameters for the second skewed normal distribution
a2, loc2, scale2 = 5, 1.5, 1  # negative a for left skew

# Generate points on the x axis
x = np.linspace(-5, 5, 1000)

# Calculate the PDFs for both distributions
pdf1 = skewnorm.pdf(x, a1, loc1, scale1)
pdf2 = skewnorm.pdf(x, a2, loc2, scale2)

# Create a mixture of the two PDFs
pdf_mixture = 0.5 * pdf1 + 0.5 * pdf2  # Assuming equal weight for simplicity

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_mixture, label="Mixture of two skewed normal distributions")
plt.fill_between(x, pdf_mixture, alpha=0.5)
plt.xlim(-2, 5)
plt.title("PDF with Two Modes and Skewness")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.axhline(y=0.07, color="r", linestyle="--")
plt.legend()
plt.show()
from scipy.interpolate import interp1d

# Interpolate the mixture PDF to find exact points of intersection
f = interp1d(x, pdf_mixture - 0.07)  # Subtract 0.07 to find roots

# Since finding exact zeros of the function is complex and depends on the function shape,
# we'll use a simple approach to find changes in sign, which indicate crossing points.
crossing_points = x[np.where(np.diff(np.sign(f(x))) != 0)[0]]

# Plotting with vertical lines for intersections
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_mixture, label="Mixture of two skewed normal distributions")
plt.fill_between(x, pdf_mixture, alpha=0.5)
plt.xlim(-2, 5)
plt.ylim(0, 0.4)
plt.title("PDF with Two Modes and Skewness")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.axhline(y=0.07, color="r", linestyle="--")

# Draw vertical lines at the crossing points
for cp in crossing_points:
    plt.axvline(x=cp, color="g", linestyle="--")

plt.legend()
plt.show()

crossing_points

# %%
from scipy.integrate import cumtrapz

# Calculate the CDF by numerical integration
cdf = cumtrapz(pdf_mixture, x, initial=0)
cdf /= cdf[-1]  # Normalize to make the final value close to 1

# Interpolate to find the quantiles
f_cdf = interp1d(cdf, x)

# Calculate the 0.025 and 0.975 quantiles
quantile_0025 = f_cdf(0.025)
quantile_0975 = f_cdf(0.975)

# Plotting with quantile points
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_mixture, label="Mixture of two skewed normal distributions")
plt.fill_between(x, pdf_mixture, alpha=0.5)
plt.xlim(-2, 5)
plt.title("PDF with Two Modes and Skewness")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.axhline(y=0.07, color="r", linestyle="--")

# Draw vertical lines at the crossing points
for cp in crossing_points:
    plt.axvline(x=cp, color="g", linestyle="--")


plt.scatter(
    [-1.6, 3.2],
    [0.0, 0.0],
    color="tab:red",
    zorder=5,
    label="Interval from partial id condition",
)
# Plot quantile points
plt.scatter(
    [quantile_0025, quantile_0975],
    [0.0, 0.0],
    color="tab:blue",
    zorder=5,
    label="Quantiles 0.025 & 0.975",
)
plt.legend(loc="upper right")
plt.savefig("simulation-results/plot_density.pdf")
plt.show()

quantile_0025, quantile_0975
# %%
