# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm

# Adjust the ranges for YL and YU for zooming in on the range -2 to 2
yl_range_zoomed = np.linspace(-2, 2, 100)
yu_range_zoomed = np.linspace(-2, 2, 100)

# Create a grid for YL and YU within the zoomed range
YL_zoomed, YU_zoomed = np.meshgrid(yl_range_zoomed, yu_range_zoomed)

# Calculate Y from YL and YU as Y = (YL + YU) / 2, given YL = Y - Z1 and YU = Y + Z2, for the zoomed range
Y_zoomed = (YL_zoomed + YU_zoomed) / 2

# Calculate Z1 and Z2 from Y, YL, and YU for the zoomed range
Z1_zoomed = Y_zoomed - YL_zoomed
Z2_zoomed = YU_zoomed - Y_zoomed

# Compute the density for the zoomed range
joint_density_zoomed = norm.pdf(Y_zoomed) * expon.pdf(Z1_zoomed) * expon.pdf(Z2_zoomed)

# Plot for the zoomed range
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(
    YL_zoomed, YU_zoomed, joint_density_zoomed, levels=100, cmap="cividis"
)
fig.colorbar(contour)

ax.plot([-0.75, -0.75], [-0.75, 0.5], color="tab:red", linestyle="-", alpha=0.7)
ax.plot([-0.75, 0.5], [0.5, 0.5], color="tab:red", linestyle="-", alpha=0.7)

ax.set_title("Joint Density of $Y_L$ and $Y_U$")
ax.set_xlabel("$Y_L$")
ax.set_ylabel("$Y_U$")
ax.set_facecolor("white")  # Ensuring the background color is white
plt.savefig("simulation-results/joint-density.pdf")
plt.show()

# %%
