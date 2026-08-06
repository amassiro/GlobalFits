"""
Uses numpy / scipy / matplotlib instead of TF1/TF2/TCanvas/TMath.

Run with:  python3 eft.py
Figures are saved as PNG files in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import brentq, minimize


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def mu_1op(x, sm, lin, quad):
    """Expected yield as a function of one EFT coefficient c (== 'x')."""
    return sm + lin * x + quad * x**2


def mu_2op(x, y, sm, lin_a, quad_a, lin_b, quad_b, int_ab):
    """Expected yield as a function of two EFT coefficients cA ('x'), cB ('y')."""
    return sm + lin_a * x + quad_a * x**2 + lin_b * y + quad_b * y**2 + int_ab * x * y


def poisson_m2logL(n_obs, mu):
    """-2 * log( Poisson(n_obs | mu) ), equivalent to TMath::Poisson based TF1."""
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-12, None)  # avoid log(0) the same way ROOT effectively does
    return -2.0 * (n_obs * np.log(mu) - mu - gammaln(n_obs + 1.0))


def find_crossings(f, x_min_search, x_max_search, x_at_min, level, n_scan=2000):
    """
    Find the two x values where f(x) - level == 0, bracketing the minimum
    at x_at_min (mirrors f_delta->GetX(level, min, x_min) / GetX(level, x_min, max)).
    """
    xs = np.linspace(x_min_search, x_max_search, n_scan)
    vals = f(xs) - level

    def root_in_range(xlo, xhi):
        sub = xs[(xs >= xlo) & (xs <= xhi)]
        v = f(sub) - level
        sign_changes = np.where(np.diff(np.sign(v)) != 0)[0]
        if len(sign_changes) == 0:
            return None
        i = sign_changes[0]
        return brentq(lambda xx: f(xx) - level, sub[i], sub[i + 1])

    low = root_in_range(x_min_search, x_at_min)
    high = root_in_range(x_at_min, x_max_search)
    return low, high


# ----------------------------------------------------------------------
# 1) -2 log Likelihood, single measurement, single operator
# ----------------------------------------------------------------------

expected_yield_SM = 12.5
expected_yield_Lin = -1.0
expected_yield_Quad = 0.1

xmin, xmax = -20, 40
N_measured = 14

print(f"mu_definition = {expected_yield_SM} + x*({expected_yield_Lin}) + x*x*({expected_yield_Quad})")

def m2logL(x):
    return poisson_m2logL(N_measured, mu_1op(x, expected_yield_SM, expected_yield_Lin, expected_yield_Quad))

x_grid = np.linspace(xmin, xmax, 100)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x_grid, m2logL(x_grid))
ax1.set_xlabel("c")
ax1.set_ylabel("-2 log P(N,c)")
ax1.set_title("-2 * Log Likelihood")
fig1.savefig("c4_m2LogLikelihood.png", dpi=150)

# minimum of -2logL
res = minimize(lambda p: m2logL(p[0]), x0=[0.0], bounds=[(xmin, xmax)])
x_min = res.x[0]
y_min = m2logL(x_min)
print(f"x_min = {x_min}")
print(f"y_min = {y_min}")

def m2logL_shifted(x):
    return m2logL(x) - y_min

min_x_draw, max_x_draw = -10, 20
x_draw = np.linspace(min_x_draw, max_x_draw, 400)

mu_low, mu_high = find_crossings(m2logL_shifted, xmin, xmax, x_min, 1.0)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(x_draw, m2logL_shifted(x_draw), color="tab:blue", lw=2)
ax2.axhline(1, color="red", lw=2)
ax2.axhline(4, color="red", lw=2)
if mu_low is not None:
    ax2.plot([mu_low, mu_low], [0, 1.0], color="red", lw=3, linestyle=":")
if mu_high is not None:
    ax2.plot([mu_high, mu_high], [0, 1.0], color="red", lw=3, linestyle=":")
ax2.set_xlabel("c")
ax2.set_ylabel("-2 log P(N,c)")
ax2.set_title("-2 * Log Likelihood (shifted)")
ax2.set_xlim(min_x_draw, max_x_draw)
ax2.grid(True)
fig2.savefig("c5_m2LogLikelihood_shifted.png", dpi=150)


# ----------------------------------------------------------------------
# 2) Build a chi2, single measurement, single operator
# ----------------------------------------------------------------------

def chi2_1op(x, sm, lin, quad, n_obs):
    mu = mu_1op(x, sm, lin, quad)
    return (mu - n_obs)**2 / mu

def chi2(x):
    return chi2_1op(x, expected_yield_SM, expected_yield_Lin, expected_yield_Quad, N_measured)

res = minimize(lambda p: chi2(p[0]), x0=[0.0], bounds=[(xmin, xmax)])
x_min_chi2 = res.x[0]

mu_low_chi2, mu_high_chi2 = find_crossings(chi2, xmin, xmax, x_min_chi2, 1.0)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(x_draw, chi2(x_draw), color="tab:blue", lw=2)
ax3.axhline(1, color="red", lw=2)
ax3.axhline(4, color="red", lw=2)
if mu_low_chi2 is not None:
    ax3.plot([mu_low_chi2, mu_low_chi2], [0, 1.0], color="red", lw=3, linestyle=":")
if mu_high_chi2 is not None:
    ax3.plot([mu_high_chi2, mu_high_chi2], [0, 1.0], color="red", lw=3, linestyle=":")
ax3.set_xlabel("c")
ax3.set_ylabel(r"$\chi^2(N,c)$")
ax3.set_title(r"$\chi^2$")
ax3.set_xlim(min_x_draw, max_x_draw)
ax3.grid(True)
fig3.savefig("c8_chi2.png", dpi=150)


# ----------------------------------------------------------------------
# 3) 2D: two operators, single measurement
# ----------------------------------------------------------------------

expected_yield_Lin_A = -1.0
expected_yield_Quad_A = 0.1
expected_yield_Lin_B = 0.3
expected_yield_Quad_B = 0.05
expected_yield_Int_AB = 0.01

def chi2_2D(x, y):
    mu = mu_2op(x, y, expected_yield_SM,
                expected_yield_Lin_A, expected_yield_Quad_A,
                expected_yield_Lin_B, expected_yield_Quad_B,
                expected_yield_Int_AB)
    return (mu - N_measured)**2 / mu

min_x, max_x = -20, 20
min_y, max_y = -20, 20

xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100))
zz = chi2_2D(xx, yy)

res2d = minimize(lambda p: chi2_2D(p[0], p[1]), x0=[0.0, 0.0],
                  bounds=[(min_x, max_x), (min_y, max_y)])
x_min2, y_min2 = res2d.x
z_min2 = chi2_2D(x_min2, y_min2)
print(" ---- ")
print(f"minimum [{x_min2} , {y_min2}] = {z_min2}")
print(" ---- ")

fig4, ax4 = plt.subplots(figsize=(8, 6))
levels_colz = np.linspace(0, 30, 100)
cf = ax4.contourf(xx, yy, np.clip(zz, 0, 30), levels=levels_colz, cmap="viridis")
fig4.colorbar(cf, ax=ax4, label=r"$\chi^2$")
ax4.contour(xx, yy, zz, levels=[2.30], colors="red", linewidths=3)  # 68% CL contour (1 dof-style)
ax4.set_xlabel("cA")
ax4.set_ylabel("cB")
ax4.set_title(r"$\chi^2(N,c_A,c_B)$")
ax4.set_xlim(-10, 20)
ax4.set_ylim(-10, 20)
fig4.savefig("c9_chi2_2D.png", dpi=150)

fig5 = plt.figure(figsize=(8, 6))
ax5 = fig5.add_subplot(111, projection="3d")
ax5.plot_surface(xx, yy, np.clip(zz, 0, 10), cmap="viridis")
ax5.set_xlabel("cA")
ax5.set_ylabel("cB")
ax5.set_zlabel(r"$\chi^2$")
fig5.savefig("c9_chi2_2D_cont.png", dpi=150)


# ----------------------------------------------------------------------
# 4) chi2 for 2 experiments/bins, single operator
# ----------------------------------------------------------------------

expected_yield_alpha_SM = 12.5
expected_yield_alpha_Lin = -1.0
expected_yield_alpha_Quad = 0.1
N_measured_alpha = 14

expected_yield_beta_SM = 22.5
expected_yield_beta_Lin = 1.5
expected_yield_beta_Quad = 0.1
N_measured_beta = 19

xmin2, xmax2 = -20, 20

def chi2_alpha(x):
    return chi2_1op(x, expected_yield_alpha_SM, expected_yield_alpha_Lin,
                     expected_yield_alpha_Quad, N_measured_alpha)

def chi2_beta(x):
    return chi2_1op(x, expected_yield_beta_SM, expected_yield_beta_Lin,
                     expected_yield_beta_Quad, N_measured_beta)

def chi2_alpha_beta(x):
    return chi2_alpha(x) + chi2_beta(x)

res_ab = minimize(lambda p: chi2_alpha_beta(p[0]), x0=[0.0], bounds=[(xmin2, xmax2)])
x_min_ab = res_ab.x[0]
y_min_ab = chi2_alpha_beta(x_min_ab)

def chi2_alpha_beta_shifted(x):
    return chi2_alpha_beta(x) - y_min_ab

x_draw2 = np.linspace(xmin2, xmax2, 400)

fig6, axs6 = plt.subplots(1, 3, figsize=(15, 4))

axs6[0].plot(x_draw2, chi2_alpha_beta_shifted(x_draw2), label=r"$\alpha+\beta$", color="tab:orange")
axs6[0].plot(x_draw2, chi2_alpha(x_draw2), label=r"$\alpha$", color="tab:blue")
axs6[0].plot(x_draw2, chi2_beta(x_draw2), label=r"$\beta$", color="teal")
axs6[0].set_ylim(0, 10)
axs6[0].legend()
axs6[0].grid(True)
axs6[0].set_title(r"$\chi^2$ combined")

axs6[1].plot(x_draw2, chi2_alpha(x_draw2), color="tab:blue", label=r"$\alpha$")
axs6[1].set_ylim(0, 10)
axs6[1].legend()
axs6[1].grid(True)
axs6[1].set_title(r"$\chi^2$ $\alpha$")

axs6[2].plot(x_draw2, chi2_beta(x_draw2), color="teal", label=r"$\beta$")
axs6[2].set_ylim(0, 10)
axs6[2].legend()
axs6[2].grid(True)
axs6[2].set_title(r"$\chi^2$ $\beta$")

for a in axs6:
    a.set_xlabel("c")
    a.set_ylabel(r"$\chi^2(N,c)$")

fig6.tight_layout()
fig6.savefig("c10_chi2_alpha_beta.png", dpi=150)


# ----------------------------------------------------------------------
# 5) 2D: two operators and two experiments combined
# ----------------------------------------------------------------------

min_x, max_x = -20, 20
min_y, max_y = -20, 20

expected_yield_alpha_SM = 12.5
expected_yield_alpha_Lin_A = -1.0
expected_yield_alpha_Quad_A = 0.1
expected_yield_alpha_Lin_B = 0.3
expected_yield_alpha_Quad_B = 0.05
expected_yield_alpha_Int_AB = 0.01
N_measured_alpha = 14

expected_yield_beta_SM = 22.5
expected_yield_beta_Lin_A = 1.5
expected_yield_beta_Quad_A = 0.1
expected_yield_beta_Lin_B = -0.1
expected_yield_beta_Quad_B = 0.03
expected_yield_beta_Int_AB = 0.1
N_measured_beta = 18

def mu_alpha_2d(x, y):
    return mu_2op(x, y, expected_yield_alpha_SM,
                  expected_yield_alpha_Lin_A, expected_yield_alpha_Quad_A,
                  expected_yield_alpha_Lin_B, expected_yield_alpha_Quad_B,
                  expected_yield_alpha_Int_AB)

def mu_beta_2d(x, y):
    return mu_2op(x, y, expected_yield_beta_SM,
                  expected_yield_beta_Lin_A, expected_yield_beta_Quad_A,
                  expected_yield_beta_Lin_B, expected_yield_beta_Quad_B,
                  expected_yield_beta_Int_AB)

def chi2_alpha_2D(x, y):
    mu = mu_alpha_2d(x, y)
    return (mu - N_measured_alpha)**2 / mu

def chi2_beta_2D(x, y):
    mu = mu_beta_2d(x, y)
    return (mu - N_measured_beta)**2 / mu

def chi2_alpha_beta_2D(x, y):
    return chi2_alpha_2D(x, y) + chi2_beta_2D(x, y)

xx2, yy2 = np.meshgrid(np.linspace(min_x, max_x, 150), np.linspace(min_y, max_y, 150))

zz_alpha_beta = chi2_alpha_beta_2D(xx2, yy2)
zz_alpha = chi2_alpha_2D(xx2, yy2)
zz_beta = chi2_beta_2D(xx2, yy2)

res_ab2 = minimize(lambda p: chi2_alpha_beta_2D(p[0], p[1]), x0=[0.0, 0.0],
                    bounds=[(min_x, max_x), (min_y, max_y)])
x_min3, y_min3 = res_ab2.x
z_min3 = chi2_alpha_beta_2D(x_min3, y_min3)
print(" ---- ")
print(f"minimum [{x_min3} , {y_min3}] = {z_min3}")
print(" ---- ")

zz_alpha_beta_shifted = zz_alpha_beta - z_min3

levels = [2.30]

fig7, ax7 = plt.subplots(figsize=(8, 6))
cf7 = ax7.contourf(xx2, yy2, np.clip(zz_alpha_beta_shifted, 0, 30),
                    levels=np.linspace(0, 30, 100), cmap="viridis")
fig7.colorbar(cf7, ax=ax7, label=r"$\chi^2$")
ax7.contour(xx2, yy2, zz_alpha_beta_shifted, levels=levels, colors="red", linewidths=3)

# individual-experiment 68% CL contours (each shifted to its own minimum), overlaid dashed
res_a2 = minimize(lambda p: chi2_alpha_2D(p[0], p[1]), x0=[0.0, 0.0],
                   bounds=[(min_x, max_x), (min_y, max_y)])
z_min_a = chi2_alpha_2D(*res_a2.x)
res_b2 = minimize(lambda p: chi2_beta_2D(p[0], p[1]), x0=[0.0, 0.0],
                   bounds=[(min_x, max_x), (min_y, max_y)])
z_min_b = chi2_beta_2D(*res_b2.x)

ax7.contour(xx2, yy2, zz_alpha - z_min_a, levels=levels, colors="darkred", linewidths=2, linestyles="dashed")
ax7.contour(xx2, yy2, zz_beta - z_min_b, levels=levels, colors="darkred", linewidths=2, linestyles="dashed")

ax7.set_xlabel("cA")
ax7.set_ylabel("cB")
ax7.set_title(r"$\chi^2(N,c_A,c_B)$ combined $\alpha+\beta$")
fig7.savefig("c11_alpha_beta_chi2_2D.png", dpi=150)

# separate alpha-only and beta-only 2D plots (mirrors c11_alpha_chi2_2D / c11_beta_chi2_2D)
for name, zz_single, zmin_single in [("alpha", zz_alpha, z_min_a), ("beta", zz_beta, z_min_b)]:
    fig, ax = plt.subplots(figsize=(8, 6))
    shifted = zz_single - zmin_single
    cf = ax.contourf(xx2, yy2, np.clip(shifted, 0, 30), levels=np.linspace(0, 30, 100), cmap="viridis")
    fig.colorbar(cf, ax=ax, label=r"$\chi^2$")
    ax.contour(xx2, yy2, shifted, levels=levels, colors="red", linewidths=3)
    ax.set_xlabel("cA")
    ax.set_ylabel("cB")
    ax.set_title(rf"$\chi^2(N,c_A,c_B)$ — {name} only")
    fig.savefig(f"c11_{name}_chi2_2D.png", dpi=150)

print("All figures saved as PNG files in the current directory.")
