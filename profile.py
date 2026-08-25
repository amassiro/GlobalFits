"""
Python re-implementation of the "c12_profile_and_2D" canvas from eft.cxx
(https://github.com/amassiro/GlobalFits), without ROOT.

Left panel:  the 2-operator, 2-experiment combined chi2(cA, cB), shifted to
             its minimum, shown as a filled map with contour lines at
             chi2 = 1, 2, ..., 10 (mirrors f_alpha_beta_chi2_2D_shifted
             drawn "colz" + "cont3 same" with many_levels).

Right panel: the profile likelihood  g(cA) = min_cB chi2(cA, cB)
             (mirrors the MinY_Wrapper / f_alpha_beta_chi2_2D_shifted_profiled
             TF1, which for every cA does a 1D numerical minimization over cB).

Run with:  python eft_profile_2D.py
Figure is saved as c12_profile_and_2D.png
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar


# ----------------------------------------------------------------------
# mu(cA, cB) for two "experiments" alpha and beta (2 operators each)
# ----------------------------------------------------------------------

def mu_2op(x, y, sm, lin_a, quad_a, lin_b, quad_b, int_ab):
    return sm + lin_a * x + quad_a * x**2 + lin_b * y + quad_b * y**2 + int_ab * x * y


# alpha
expected_yield_alpha_SM = 12.5
expected_yield_alpha_Lin_A = -1.0
expected_yield_alpha_Quad_A = 0.1
expected_yield_alpha_Lin_B = 0.3
expected_yield_alpha_Quad_B = 0.05
expected_yield_alpha_Int_AB = 0.01
N_measured_alpha = 14

# beta
expected_yield_beta_SM = 22.5
expected_yield_beta_Lin_A = 1.5
expected_yield_beta_Quad_A = 0.1
expected_yield_beta_Lin_B = -0.1
expected_yield_beta_Quad_B = 0.03
expected_yield_beta_Int_AB = 0.1
N_measured_beta = 18

min_x, max_x = -20, 20
min_y, max_y = -20, 20


def mu_alpha(x, y):
    return mu_2op(x, y, expected_yield_alpha_SM,
                  expected_yield_alpha_Lin_A, expected_yield_alpha_Quad_A,
                  expected_yield_alpha_Lin_B, expected_yield_alpha_Quad_B,
                  expected_yield_alpha_Int_AB)


def mu_beta(x, y):
    return mu_2op(x, y, expected_yield_beta_SM,
                  expected_yield_beta_Lin_A, expected_yield_beta_Quad_A,
                  expected_yield_beta_Lin_B, expected_yield_beta_Quad_B,
                  expected_yield_beta_Int_AB)


def chi2_alpha_2D(x, y):
    mu = mu_alpha(x, y)
    return (mu - N_measured_alpha) ** 2 / mu


def chi2_beta_2D(x, y):
    mu = mu_beta(x, y)
    return (mu - N_measured_beta) ** 2 / mu


def chi2_alpha_beta_2D(x, y):
    """f(x, y) -- the combined 2D chi2 (mirrors f_alpha_beta_chi2_2D)."""
    return chi2_alpha_2D(x, y) + chi2_beta_2D(x, y)


# global minimum of f(x,y), to shift the map to 0 (mirrors GetMinimumXY + Eval)
res_min = minimize(lambda p: chi2_alpha_beta_2D(p[0], p[1]), x0=[0.0, 0.0],
                    bounds=[(min_x, max_x), (min_y, max_y)])
x_min, y_min = res_min.x
z_min = chi2_alpha_beta_2D(x_min, y_min)
print(f"minimum [{x_min}, {y_min}] = {z_min}")


def f_shifted(x, y):
    """f_alpha_beta_chi2_2D_shifted: f(x,y) - z_min."""
    return chi2_alpha_beta_2D(x, y) - z_min


# ----------------------------------------------------------------------
# g(x) = min_y f_shifted(x, y)   -- the "profile" (mirrors MinY_Wrapper)
# ----------------------------------------------------------------------

def profile_min_y(x_val, ymin=min_y, ymax=max_y):
    """For a fixed x, numerically minimize f_shifted(x, y) over y."""
    res = minimize_scalar(lambda yy: f_shifted(x_val, yy),
                           bounds=(ymin, ymax), method="bounded")
    return res.fun


# ----------------------------------------------------------------------
# build the figure: left = colz + many contours, right = profile g(x)
# ----------------------------------------------------------------------

xx, yy = np.meshgrid(np.linspace(min_x, max_x, 150), np.linspace(min_y, max_y, 150))
zz_shifted = f_shifted(xx, yy)

x_profile = np.linspace(min_x, max_x, 150)
g_profile = np.array([profile_min_y(xv) for xv in x_profile])

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

# left: colz map + contour lines at chi2 = 1..10 (mirrors many_levels)
cf = ax_left.contourf(xx, yy, np.clip(zz_shifted, 0, 30),
                       levels=np.linspace(0, 30, 100), cmap="viridis")
fig.colorbar(cf, ax=ax_left, label=r"$\chi^2$")
many_levels = list(range(1, 11))  # 1,2,...,10
ax_left.contour(xx, yy, zz_shifted, levels=many_levels, colors="red", linewidths=1.5)
ax_left.set_xlabel("cA")
ax_left.set_ylabel("cB")
ax_left.set_title(r"$\chi^2(N,c_A,c_B)$")

# right: profile g(cA) = min_cB chi2(cA, cB)
ax_right.plot(x_profile, g_profile, color="red", lw=2)
ax_right.set_xlabel("cA")
ax_right.set_ylabel(r"$\min_{c_B}\ \chi^2(c_A,c_B)$")
ax_right.set_title(r"$\min_{c_B}\ \chi^2(c_A,c_B)$")
ax_right.grid(True)

fig.tight_layout()
fig.savefig("c12_profile_and_2D.png", dpi=150)

print("Saved c12_profile_and_2D.png")


