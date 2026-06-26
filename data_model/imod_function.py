# This is the "ground truth" function to test the BO.
# 4D BO model for different QW's is the ground truth model
# Will be based on gp on all the data we have

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('all_data_summarized/data_noBAI.csv')
data = data[
    (data["R MAI"] >= 0) & (data["R MAI"] <= 2.5) &
    (data["R BAAc"] >= 0) & (data["R BAAc"] <= 2.5)
].reset_index(drop=True)

feature_cols = ["Anneal Time", "Temperature", "R MAI", "R BAAc"]
qw_cols = [f"QW{i}" for i in range(1, 13)] + ["QW99"]

# Subsample for GP tractability
rng = np.random.default_rng(42)
idx = rng.choice(len(data), size=min(300, len(data)), replace=False)
data_fit = data.iloc[idx].dropna(subset=feature_cols).reset_index(drop=True)

X_fit = data_fit[feature_cols].values
scaler = StandardScaler()
X_fit_scaled = scaler.fit_transform(X_fit)

# Build 2D prediction grid in (R BAAc, R MAI) space, fixing MAI_vol/BAAc_vol at mean
n_grid = 60
r_mai_vals = np.linspace(data["R MAI"].min(), data["R MAI"].max(), n_grid)
r_baac_vals = np.linspace(data["R BAAc"].min(), data["R BAAc"].max(), n_grid)
R_BAAc_grid, R_MAI_grid = np.meshgrid(r_baac_vals, r_mai_vals)

Anneal_Time_fixed = data["Anneal Time"].mean()
Temperature_fixed = data["Temperature"].mean()

X_pred = np.column_stack([
    np.full(R_MAI_grid.size, Anneal_Time_fixed),
    np.full(R_MAI_grid.size, Temperature_fixed),
    R_MAI_grid.ravel(),
    R_BAAc_grid.ravel(),
])
X_pred_scaled = scaler.transform(X_pred)

# Fit a GP per QW and store predictions
kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel()

gp_means = {}
gp_stds = {}

print("Fitting GPs...")
for qw in qw_cols:
    mask = data_fit[qw].notna().values
    y = data_fit[qw].values[mask]
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(X_fit_scaled[mask], y)
    mu, sigma = gp.predict(X_pred_scaled, return_std=True)
    gp_means[qw] = mu.reshape(n_grid, n_grid)
    gp_stds[qw] = sigma.reshape(n_grid, n_grid)
    print(f"  {qw} done")

print("Done. Plotting...")


# --- Figure 1: GP Mean for all QW scores ---
nrows, ncols = 4, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(22, 20))
axes = axes.flatten()

for i, qw in enumerate(qw_cols):
    ax = axes[i]
    vmin, vmax = gp_means[qw].min(), gp_means[qw].max()
    cf_mean = ax.contourf(r_baac_vals, r_mai_vals, gp_means[qw], levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    mask = data_fit[qw].notna()
    ax.scatter(
        data_fit.loc[mask, "R BAAc"], data_fit.loc[mask, "R MAI"],
        c=data_fit.loc[mask, qw], cmap="viridis", vmin=vmin, vmax=vmax,
        s=12, edgecolors="white", linewidths=0.4, zorder=3
    )
    ax.set_xlabel("R BAAc", fontsize=9)
    ax.set_ylabel("R MAI", fontsize=9)
    ax.set_title(qw, fontsize=11, weight="bold")
    fig.colorbar(cf_mean, ax=ax, fraction=0.046, pad=0.04)

for j in range(len(qw_cols), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "4D GP Mean  |  inputs: Anneal Time, Temperature, R MAI, R BAAc\n"
    f"(Anneal Time = {Anneal_Time_fixed:.1f}, Temperature = {Temperature_fixed:.1f} fixed at mean)",
    fontsize=14, y=1.01
)
plt.tight_layout()
plt.savefig("4d_gp_mean.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 2: GP Uncertainty (std) for all QW scores ---
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(22, 20))
axes2 = axes2.flatten()

for i, qw in enumerate(qw_cols):
    ax = axes2[i]
    std_vmin, std_vmax = gp_stds[qw].min(), gp_stds[qw].max()
    cf_std = ax.contourf(r_baac_vals, r_mai_vals, gp_stds[qw], levels=20, cmap="plasma", vmin=std_vmin, vmax=std_vmax)
    ax.scatter(
        data_fit["R BAAc"], data_fit["R MAI"],
        c="white", s=8, alpha=0.5, zorder=3
    )
    ax.set_xlabel("R BAAc", fontsize=9)
    ax.set_ylabel("R MAI", fontsize=9)
    ax.set_title(qw, fontsize=11, weight="bold")
    fig2.colorbar(cf_std, ax=ax, fraction=0.046, pad=0.04)

for j in range(len(qw_cols), len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle(
    "4D GP Uncertainty (std)  |  inputs: Anneal Time, Temperature, R MAI, R BAAc\n"
    f"(Anneal Time = {Anneal_Time_fixed:.1f}, Temperature = {Temperature_fixed:.1f} fixed at mean)",
    fontsize=14, y=1.01
)
plt.tight_layout()
plt.savefig("4d_gp_std.png", dpi=150, bbox_inches="tight")
plt.show()


