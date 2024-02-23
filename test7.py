# %%
# info: this file tests the case when identified set is known.
import numpy as np
import matplotlib.pyplot as plt
from utils_dgp import compute_score

N = 1000
M = 500


# %%
def generate_data(N, M):
    x = np.random.uniform(0, 4, (N, 1))
    y = x + np.random.normal(0, 1, (N, 1))
    yl = np.floor(y)
    yu = yl + 1
    return x, y, yl, yu


x, y, yl, yu = generate_data(N, M)
# %%
plt.plot(x, y, "o")
plt.plot(x, yl, "o")
plt.plot(x, yu, "o")
# %%
from sklearn.kernel_ridge import KernelRidge

mm = KernelRidge(alpha=0.1, kernel="rbf")
yl_fit = mm.fit(x, yl).predict(x)
yu_fit = mm.fit(x, yu).predict(x)
plt.scatter(x, y)
plt.plot(x, x)
plt.scatter(x, yl_fit)
plt.scatter(x, yu_fit)
# %% 
plt.scatter(x, x - yl_fit)
plt.xlim(0,4)
(x-yl_fit).max()
# %%
x_test, y_test, yl_test, yu_test = generate_data(N, M)
compute_score(x_test.T, yu_test.T, yl_test.T)
compute_score(mm.fit(x, yl).predict(x).T, yu_test.T, yl_test.T)

# %%
plt.scatter(x, yl, alpha=0.5)
# %%
yl_fixed_x = np.floor(np.random.normal(0, 1, (N, 1)))
plt.hist(yl_fixed_x)
# %%
