"""
Python re-implementation of chi2nuisance.cxx (https://github.com/amassiro/GlobalFits)
without ROOT.

Physics: N ~ Poisson(mu * theta), where theta is a luminosity scale factor
constrained by a Gaussian pull term theta ~ Gaussian(mean=1.0, sigma=lumi_uncertainty).
mu is the parameter of interest, theta is a nuisance parameter.

Uses numpy / scipy / matplotlib instead of TF2/TF1/TCanvas.

Run with:  python chi2nuisance.py
Figures are saved as PNG files in the current directory.
"""

import math
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar


# ----------------------------------------------------------------------
# physics functions
# ----------------------------------------------------------------------

def log_gauss(y, mean, sigma):
    """log of the *normalized* Gaussian pdf (mirrors TMath::Gaus(y, mean, sigma, true))."""
    return -0.5 * ((y - mean) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))


def log_likelihood(mu, theta, n_obs, sigma_lumi):
    """
    log[ exp(-mu*theta) * (mu*theta)^N / N!  *  Gaus(theta; 1.0, sigma_lumi) ]
    (mirrors f_likelihood: Poisson(N | mu*theta) times a Gaussian constraint on theta)
    """
    mu = np.asarray(mu, dtype=float)
    theta = np.asarray(theta, dtype=float)
    xy = mu * theta
    lgamma_n1 = math.lgamma(n_obs + 1.0)
    with np.errstate(divide="ignore"):
        log_pois = -xy + n_obs * np.log(np.where(xy > 0, xy, 1.0)) - lgamma_n1
    log_pois = np.where(xy > 0, log_pois, -1e10)  # finite stand-in for -inf, avoids NaNs in the optimizer
    return log_pois + log_gauss(theta, 1.0, sigma_lumi)


def likelihood(mu, theta, n_obs, sigma_lumi):
    return np.exp(log_likelihood(mu, theta, n_obs, sigma_lumi))


def m2logL_not_shifted(mu, theta, n_obs, sigma_lumi):
    return -2.0 * log_likelihood(mu, theta, n_obs, sigma_lumi)


def simple_m2logL_not_shifted(mu, n_obs):
    """-2 * log( exp(-mu) * mu^N / N! ), no nuisance parameter at all."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    lgamma_n1 = math.lgamma(n_obs + 1.0)
    return -2.0 * (-mu + n_obs * np.log(mu) - lgamma_n1)


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------

expected_yield = 12.5  # unused directly, kept for parity with the original file
min_x, max_x = 0, 30
min_y, max_y = 0.5, 1.5

lumi_uncertainty = 0.02
N_measured = 14


# ----------------------------------------------------------------------
# 1) 2D Likelihood P(N, mu, theta)
# ----------------------------------------------------------------------

xx, yy = np.meshgrid(np.linspace(min_x, max_x, 150), np.linspace(min_y, max_y, 150))
zz_likelihood = likelihood(xx, yy, N_measured, lumi_uncertainty)

fig1, ax1 = plt.subplots(figsize=(8, 6))
pc1 = ax1.pcolormesh(xx, yy, zz_likelihood, shading="auto", cmap="viridis")
fig1.colorbar(pc1, ax=ax1, label=r"$P(N,\mu,\theta)$")
ax1.set_xlabel(r"$\mu$")
ax1.set_ylabel(r"$\theta$")
ax1.set_title("Likelihood")
fig1.savefig("c3_likelihood.png", dpi=150)


# ----------------------------------------------------------------------
# 2) -2 log Likelihood, shifted to its global minimum over (mu, theta)
# ----------------------------------------------------------------------

def m2logL_not_shifted_scalar(p):
    mu, theta = p
    return float(m2logL_not_shifted(mu, theta, N_measured, lumi_uncertainty))

res2d = minimize(m2logL_not_shifted_scalar, x0=[N_measured, 1.0],
                  bounds=[(min_x, max_x), (min_y, max_y)])
x_min, y_min = res2d.x
z_min = m2logL_not_shifted_scalar([x_min, y_min])
print(f" x,y,z = {x_min} , {y_min} , {z_min}")


def m2loglikelihood(mu, theta):
    """f_m2loglikelihood: shifted to 0 at the global minimum."""
    return m2logL_not_shifted(mu, theta, N_measured, lumi_uncertainty) - z_min


zz_m2logL = m2loglikelihood(xx, yy)

fig2, ax2 = plt.subplots(figsize=(8, 6))
pc2 = ax2.pcolormesh(xx, yy, np.clip(zz_m2logL, 0, 30), shading="auto",
                      cmap="viridis", vmin=0, vmax=30)
fig2.colorbar(pc2, ax=ax2, label=r"-2 log $P(N,\mu,\theta)$")
ax2.set_xlabel(r"$\mu$")
ax2.set_ylabel(r"$\theta$")
ax2.set_title("-2 log Likelihood")
fig2.savefig("c3_m2loglikelihood.png", dpi=150)


# ----------------------------------------------------------------------
# 3) Profile: g(mu) = min_theta m2loglikelihood(mu, theta)   (mirrors MinY_Wrapper)
# ----------------------------------------------------------------------

def profile_min_theta(mu_val, theta_lo=min_y, theta_hi=max_y):
    res = minimize_scalar(lambda th: m2loglikelihood(mu_val, th),
                           bounds=(theta_lo, theta_hi), method="bounded")
    return res.fun


mu_grid = np.linspace(min_x, max_x, 200)
g_profile = np.array([profile_min_theta(mv) for mv in mu_grid])


# ----------------------------------------------------------------------
# 4) Simple -2logL(mu) with NO nuisance parameter, shifted to its own minimum
# ----------------------------------------------------------------------

res_simple = minimize_scalar(lambda mv: simple_m2logL_not_shifted(mv, N_measured),
                              bounds=(min_x, max_x), method="bounded")
x_min_simple = res_simple.x
y_min_simple = simple_m2logL_not_shifted(x_min_simple, N_measured)


def simple_m2logL(mu):
    return simple_m2logL_not_shifted(mu, N_measured) - y_min_simple


# ----------------------------------------------------------------------
# 5) Overlay both curves: profiled (with nuisance) vs. simple (without)
# ----------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(mu_grid, g_profile, color="red", lw=2,
         label=r"profiled over $\theta$ (with nuisance)")
ax3.plot(mu_grid, simple_m2logL(mu_grid), color="blue", lw=2,
         label="no nuisance parameter")
ax3.set_xlabel(r"$\mu$")
ax3.set_ylabel(r"-2 log $P(N,\mu)$")
ax3.set_title("-2 log Likelihood")
ax3.set_ylim(0, 10)
ax3.legend()
ax3.grid(True)
fig3.savefig("c3_m2loglikelihood_profile.png", dpi=150)

print("All figures saved as PNG files in the current directory.")

