"""
Python re-implementation of chi2.cxx (https://github.com/amassiro/GlobalFits)
without ROOT.

Uses numpy / scipy / matplotlib. Root-finding (brentq) and minimization
(minimize_scalar) use scipy.optimize directly, rather than hand-rolled
implementations.

Run with:  python chi2.py
Figures are saved as PNG files in the current directory.
"""

import matplotlib
matplotlib.use("Agg")

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar


# ----------------------------------------------------------------------
# physics functions (gammaln(n+1) is only ever needed for a fixed integer
# n_obs here, so plain math.lgamma is enough -- no array version required)
# ----------------------------------------------------------------------

def likelihood_fixed_N(mu, n_obs):
    """exp(-mu) * mu^N / N!  as a function of mu, for fixed observed N."""
    mu = np.asarray(mu, dtype=float)
    log_fact = math.lgamma(n_obs + 1.0)
    #
    # gamma(n+1) = n!
    # lgamma = log(gamma)
    #
    with np.errstate(divide="ignore"):
        log_l = -mu + n_obs * np.log(np.where(mu > 0, mu, 1.0)) - log_fact
    return np.where(mu > 0, np.exp(log_l), 0.0)


def m2logL_fixed_N(mu, n_obs):
    """-2 * log( exp(-mu) * mu^N / N! ) as a function of mu."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    #
    # set minimum to 10^-12
    #
    log_fact = math.lgamma(n_obs + 1.0)
    return -2.0 * (-mu + n_obs * np.log(mu) - log_fact)


def chi2_fixed_N(mu, n_obs):
    """(mu - N)^2 / mu"""
    mu = np.asarray(mu, dtype=float)
    mu_safe = np.where(mu == 0, 1e-12, mu)
    return (mu - n_obs) ** 2 / mu_safe


def find_crossings(f, x_lo, x_hi, x_at_min, level, n_scan=4000):
    """Find the two roots of f(x) - level = 0 bracketing x_at_min (mirrors TF1::GetX),
    using scipy.optimize.brentq for the actual root-finding."""
    xs = np.linspace(x_lo, x_hi, n_scan)
    vals = f(xs) - level

    def root_in_range(a_idx, b_idx):
        sign = np.sign(vals[a_idx:b_idx])
        #
        # +1, 0, -1 --> the vector of signs
        #
        idx = np.where(np.diff(sign) != 0)[0]
        #
        # first index of when there is a sign flip
        #
        if len(idx) == 0:
            return None
        i = a_idx + idx[0]
        return brentq(lambda xx: f(xx) - level, xs[i], xs[i + 1])
        #
        # return the value for which f(xx) - level == 0, in the interval between xs[i] and xs[i + 1]
        #

    mid = int(np.searchsorted(xs, x_at_min))
    low = root_in_range(0, mid)
    high = root_in_range(mid, len(xs) - 1)
    return low, high


# ----------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------

xmin, xmax = 0, 30
N_measured = 14

min_x_draw, max_x_draw = 5, 25
mu_grid = np.linspace(xmin, xmax, 400)
x_draw = np.linspace(min_x_draw, max_x_draw, 400)


# ----------------------------------------------------------------------
# 1) Likelihood, fixed measured N, as a function of mu
# ----------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(mu_grid, likelihood_fixed_N(mu_grid, N_measured))
ax3.set_xlabel(r"$\mu$")
ax3.set_ylabel(r"$P(N,\mu)$")
ax3.set_title("Likelihood")
fig3.savefig("c3_likelihood.png", dpi=150)


# ----------------------------------------------------------------------
# 2) -2 log Likelihood
# ----------------------------------------------------------------------

def m2logL(mu):
    return m2logL_fixed_N(mu, N_measured)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(mu_grid, m2logL(mu_grid))
ax4.set_xlabel(r"$\mu$")
ax4.set_ylabel(r"-2 log $P(N,\mu)$")
ax4.set_title("-2 * Log Likelihood")
fig4.savefig("c4_m2LogLikelihood.png", dpi=150)

res = minimize_scalar(m2logL, bounds=(xmin, xmax), method="bounded")
x_min = res.x
y_min = m2logL(x_min)


# ----------------------------------------------------------------------
# 3) -2 log Likelihood, shifted, with 1sigma/2sigma (Delta = 1, 4) lines
# ----------------------------------------------------------------------

def m2logL_shifted(mu):
    return m2logL(mu) - y_min

x_min = minimize_scalar(m2logL_shifted, bounds=(xmin, xmax), method="bounded").x
mu_low, mu_high = find_crossings(m2logL_shifted, xmin, xmax, x_min, 1.0)

fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.plot(x_draw, m2logL_shifted(x_draw), color="tab:blue", lw=2)
ax5.axhline(1, color="red", lw=2)
ax5.axhline(4, color="red", lw=2)
if mu_low is not None:
    ax5.plot([mu_low, mu_low], [0, 1.0], color="red", lw=3, linestyle=":")
if mu_high is not None:
    ax5.plot([mu_high, mu_high], [0, 1.0], color="red", lw=3, linestyle=":")
ax5.set_xlabel(r"$\mu$")
ax5.set_ylabel(r"-2 log $P(N,\mu)$")
ax5.set_title("-2 * Log Likelihood (shifted)")
ax5.set_xlim(min_x_draw, max_x_draw)
ax5.grid(True)
fig5.savefig("c5_m2LogLikelihood_shifted.png", dpi=150)


# ----------------------------------------------------------------------
# 4) Build a chi2 = (mu - N)^2 / mu
# ----------------------------------------------------------------------

def chi2(mu):
    return chi2_fixed_N(mu, N_measured)

x_min = minimize_scalar(chi2, bounds=(xmin, xmax), method="bounded").x
y_min = chi2(x_min)
mu_low, mu_high = find_crossings(chi2, xmin, xmax, x_min, 1.0)

print("chi2")
print(f"x_min = {x_min}")
print(f"y_min = {y_min}")

fig8, ax8 = plt.subplots(figsize=(8, 6))
ax8.plot(x_draw, chi2(x_draw), color="tab:blue", lw=2)
ax8.axhline(1, color="red", lw=2)
ax8.axhline(4, color="red", lw=2)
if mu_low is not None:
    ax8.plot([mu_low, mu_low], [0, 1.0], color="red", lw=3, linestyle=":")
if mu_high is not None:
    ax8.plot([mu_high, mu_high], [0, 1.0], color="red", lw=3, linestyle=":")
ax8.set_xlabel(r"$\mu$")
ax8.set_ylabel(r"$\chi^2(N,\mu)$")
ax8.set_title(r"$\chi^2$")
ax8.set_xlim(min_x_draw, max_x_draw)
ax8.grid(True)
fig8.savefig("c8_chi2.png", dpi=150)


# ----------------------------------------------------------------------
# 5) chi2, shifted to its own minimum, with only the Delta = 1 line
# ----------------------------------------------------------------------

def chi2_shifted(mu):
    return chi2(mu) - y_min

fig9, ax9 = plt.subplots(figsize=(8, 6))
ax9.plot(x_draw, chi2_shifted(x_draw), color="tab:blue", lw=2)
ax9.axhline(1, color="red", lw=2)
ax9.set_xlabel(r"$\mu$")
ax9.set_ylabel(r"$\chi^2(N,\mu)$")
ax9.set_title(r"$\chi^2$ (shifted)")
ax9.set_xlim(min_x_draw, max_x_draw)
ax9.grid(True)
fig9.savefig("c8_chi2_shifted.png", dpi=150)

print("All figures saved as PNG files in the current directory.")


