"""
Python re-implementation of poisson.cxx (https://github.com/amassiro/GlobalFits)
without any dependency on ROOT.

Uses numpy / scipy / matplotlib instead of TF1/TH1F/TH2F/TRandom3/TCanvas.

Run with:  python3 poisson.py
Figures are saved as PNG files in the current directory.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import gammaln
from scipy.optimize import brentq, minimize_scalar


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def poisson_pmf(x, mu):
    """
    Equivalent of ROOT's TMath::Poisson(x, mu) = mu^x * exp(-mu) / Gamma(x+1).
    Works for continuous, non-negative x (as ROOT's TF1 does), not just integers.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    x_b, mu_b = np.broadcast_arrays(x, mu)
    out = np.zeros(x_b.shape)

    zero_mu = (mu_b == 0)
    zero_x_at_zero_mu = zero_mu & (x_b == 0)
    normal = ~zero_mu

    out = np.where(zero_x_at_zero_mu, 1.0, out)
    with np.errstate(divide="ignore"):
        log_pmf = x_b * np.log(np.where(normal, mu_b, 1.0)) - mu_b - gammaln(x_b + 1.0)
    out = np.where(normal, np.exp(log_pmf), out)
    return out


def likelihood_fixed_N(mu, n_obs):
    """exp(-mu) * mu^N / N!  as a function of mu, for fixed observed N (mirrors f_likelihood)."""
    mu = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore"):
        log_l = -mu + n_obs * np.log(np.where(mu > 0, mu, 1.0)) - gammaln(n_obs + 1.0)
    return np.where(mu > 0, np.exp(log_l), 0.0)


def m2logL_fixed_N(mu, n_obs):
    """-2 * log( exp(-mu) * mu^N / N! ) as a function of mu (mirrors f_m2LogLikelihood)."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    return -2.0 * (-mu + n_obs * np.log(mu) - gammaln(n_obs + 1.0))


def chi2_fixed_N(mu, n_obs):
    """(mu - N)^2 / mu (mirrors f_chi2)."""
    mu = np.asarray(mu, dtype=float)
    mu_safe = np.where(mu == 0, 1e-12, mu)
    return (mu - n_obs) ** 2 / mu_safe


def find_crossings(f, x_lo, x_hi, x_at_min, level, n_scan=4000):
    """Find the two roots of f(x) - level = 0 bracketing x_at_min."""
    xs = np.linspace(x_lo, x_hi, n_scan)

    def root_in_range(a, b):
        sub = xs[(xs >= a) & (xs <= b)]
        v = f(sub) - level
        idx = np.where(np.diff(np.sign(v)) != 0)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        return brentq(lambda xx: f(xx) - level, sub[i], sub[i + 1])

    low = root_in_range(x_lo, x_at_min)
    high = root_in_range(x_at_min, x_hi)
    return low, high


rng = np.random.default_rng(0)  # equivalent of TRandom3 rng(0)


# ----------------------------------------------------------------------
# 1) Poisson distribution, single curve
# ----------------------------------------------------------------------

expected_yield = 12.5
xmin, xmax = 0, 30

x_grid = np.linspace(xmin, xmax, 400)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x_grid, poisson_pmf(x_grid, expected_yield))
ax1.set_xlabel("N")
ax1.set_ylabel("P(N)")
ax1.set_title("Poisson Distribution")
fig1.savefig("c1_function.png", dpi=150)


# ----------------------------------------------------------------------
# 2) Many Poisson curves for different mu, colored like a palette
# ----------------------------------------------------------------------

num_of_curves = 20
cmap = matplotlib.colormaps.get_cmap("cool").resampled(num_of_curves)

fig2, ax2 = plt.subplots(figsize=(8, 6))
for ii in range(num_of_curves):
    mu = ii * 0.5
    ax2.plot(x_grid, poisson_pmf(x_grid, mu), color=cmap(ii), label=rf"$\mu$ = {mu:.1f}")
ax2.set_xlabel("N")
ax2.set_ylabel("P(N)")
ax2.set_title("Poisson Functions")
ax2.legend(fontsize=7, loc="upper right", ncol=2)
ax2.grid(True)
fig2.savefig("c1_many_function.png", dpi=150)


# ----------------------------------------------------------------------
# 3) Generate toy events ~ Poisson(expected_yield)
# ----------------------------------------------------------------------

nEvents = 10
nbins = int(xmax - xmin)

events = rng.poisson(expected_yield, size=nEvents)
average_events = events.mean()

print(f"expected_yield = {expected_yield}")
print(f"average toys = {average_events}")

fig3, ax3 = plt.subplots(figsize=(8, 6))
bin_edges = np.arange(xmin - 0.5, xmax + 0.5, 1.0)
ax3.hist(events, bins=bin_edges, color="tab:blue", alpha=0.4, edgecolor="tab:blue", linewidth=2)
ax3.set_xlabel("n")
ax3.set_ylabel("Entries")
ax3.set_title("Sampled Poisson")
fig3.savefig("c2_generate_events.png", dpi=150)


# ----------------------------------------------------------------------
# 4) Likelihood, fixed measured N, as function of mu
# ----------------------------------------------------------------------

N_measured = 14

mu_grid = np.linspace(xmin, xmax, 400)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(mu_grid, likelihood_fixed_N(mu_grid, N_measured))
ax4.set_xlabel(r"$\mu$")
ax4.set_ylabel(r"$P(N,\mu)$")
ax4.set_title("Likelihood")
fig4.savefig("c3_likelihood.png", dpi=150)


# ----------------------------------------------------------------------
# 5) -2 log Likelihood, minimum, shifted, 1sigma/2sigma crossings
# ----------------------------------------------------------------------

def m2logL(mu):
    return m2logL_fixed_N(mu, N_measured)

res = minimize_scalar(m2logL, bounds=(xmin, xmax), method="bounded")
x_min = res.x
y_min = m2logL(x_min)

def m2logL_shifted(mu):
    return m2logL(mu) - y_min

min_x_draw, max_x_draw = 5, 25
x_draw = np.linspace(min_x_draw, max_x_draw, 400)

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
# 6) Toy MC: what does a 68% CL interval mean? Build a "confidence belt" of toys
# ----------------------------------------------------------------------

n_toys = 100
nbins_fine = nbins * 10
mu_fine_edges = np.linspace(xmin - 0.5, xmax - 0.5, nbins_fine + 1)
delta_xx = (xmax - xmin) / nbins_fine
print(f" delta_xx = {delta_xx}")

h_mu_best = np.zeros(n_toys)
h_mu_low = np.zeros(n_toys)
h_mu_high = np.zeros(n_toys)

# support map: rows = toy index, columns = mu bins -> 1 where mu is inside that toy's 1sigma interval
h_support = np.zeros((n_toys, nbins_fine))

for itoy in range(n_toys):
    event_toy = rng.poisson(expected_yield)

    def nll(mu, n_obs=event_toy):
        mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
        return -2.0 * np.log(np.clip(poisson_pmf(n_obs, mu), 1e-300, None))

    res_toy = minimize_scalar(nll, bounds=(xmin, xmax), method="bounded")
    x_min_toy = res_toy.x
    y_min_toy = nll(x_min_toy)
    target_y = y_min_toy + 1.0

    if x_min_toy > xmin:
        lo, _ = find_crossings(nll, xmin, xmax, x_min_toy, target_y)
        min_x_toy = lo if lo is not None else 0.0
    else:
        min_x_toy = 0.0
    _, hi = find_crossings(nll, xmin, xmax, x_min_toy, target_y)
    max_x_toy = hi if hi is not None else xmax

    h_mu_best[itoy] = x_min_toy
    h_mu_low[itoy] = min_x_toy
    h_mu_high[itoy] = max_x_toy

    col_lo = int(round((min_x_toy - (xmin - 0.5)) / delta_xx))
    col_hi = int(round((max_x_toy - (xmin - 0.5)) / delta_xx))
    col_lo = max(0, min(nbins_fine - 1, col_lo))
    col_hi = max(0, min(nbins_fine - 1, col_hi))
    h_support[itoy, col_lo:col_hi + 1] = 1.0

fig6, ax6 = plt.subplots(figsize=(8, 6))
extent = [mu_fine_edges[0], mu_fine_edges[-1], 0, n_toys]
ax6.imshow(h_support, aspect="auto", origin="lower", extent=extent, cmap="viridis")
ax6.set_xlabel(r"$\mu$")
ax6.set_ylabel("Toys")
ax6.set_title("Toys confidence belt")
ax6.grid(True, alpha=0.3)
fig6.savefig("c6_toys_belt.png", dpi=150)


# ----------------------------------------------------------------------
# 7) Confidence belt (Neyman construction): P(N | mu) as a 2D map
# ----------------------------------------------------------------------

nbins_mu = nbins * 40
delta_yy = (xmax - xmin) / (nbins_mu + 1)

N_vals = np.arange(0, nbins)  # integer N bins, like the ROOT TH2F x-axis
mu_vals = xmin + delta_yy * np.arange(nbins_mu)

NN, MM = np.meshgrid(N_vals, mu_vals)
belt = poisson_pmf(NN, MM)

fig7, ax7 = plt.subplots(figsize=(8, 6))
pc = ax7.pcolormesh(N_vals, mu_vals, belt, shading="auto", cmap="viridis")
fig7.colorbar(pc, ax=ax7, label="P(N|mu)")
ax7.set_xlabel("N")
ax7.set_ylabel(r"$\mu$")
ax7.set_title("Confidence belt")
ax7.grid(True, alpha=0.3)
fig7.savefig("c7_belt.png", dpi=150)


# ----------------------------------------------------------------------
# 8) Build a chi2 = (mu - N)^2 / mu
# ----------------------------------------------------------------------

def chi2(mu):
    return chi2_fixed_N(mu, N_measured)

res_chi2 = minimize_scalar(chi2, bounds=(xmin, xmax), method="bounded")
x_min_chi2 = res_chi2.x

mu_low_chi2, mu_high_chi2 = find_crossings(chi2, xmin, xmax, x_min_chi2, 1.0)

fig8, ax8 = plt.subplots(figsize=(8, 6))
ax8.plot(x_draw, chi2(x_draw), color="tab:blue", lw=2)
ax8.axhline(1, color="red", lw=2)
ax8.axhline(4, color="red", lw=2)
if mu_low_chi2 is not None:
    ax8.plot([mu_low_chi2, mu_low_chi2], [0, 1.0], color="red", lw=3, linestyle=":")
if mu_high_chi2 is not None:
    ax8.plot([mu_high_chi2, mu_high_chi2], [0, 1.0], color="red", lw=3, linestyle=":")
ax8.set_xlabel(r"$\mu$")
ax8.set_ylabel(r"$\chi^2(N,\mu)$")
ax8.set_title(r"$\chi^2$")
ax8.set_xlim(min_x_draw, max_x_draw)
ax8.grid(True)
fig8.savefig("c8_chi2.png", dpi=150)

print("All figures saved as PNG files in the current directory.")
