import numpy as np
import matplotlib.pyplot as plt
import pylhe

LHE_FILE = "unweighted_events.lhe"
PT_BINS = np.array([20, 30, 40, 50, 65, 80, 100, 130, 170, 220, 300, 500])
LUMI_FB = 300.0

BENCHMARKS = [
    ("sm", 0.0, 0.0),
    ("cHl3_p1", 1.0, 0.0),
    ("cHl3_m1", -1.0, 0.0),
    ("cll1_p1", 0.0, 1.0),
    ("cll1_m1", 0.0, -1.0),
    ("cll1_cHl3_pp", 1.0, 1.0),
]


def leading_lepton_pt(event):
    pts = [np.hypot(p.px, p.py) for p in event.particles if p.status == 1 and abs(p.id) in (11, 13, 15)]
    return max(pts) if pts else None


def read_cross_sections(lhe_file, bins):
    nbins = len(bins) - 1
    sumw = {name: np.zeros(nbins) for name, _, _ in BENCHMARKS}
    n_total = 0
    for event in pylhe.LHEFile.fromfile(lhe_file).events:
        n_total += 1
        pt = leading_lepton_pt(event)
        if pt is None:
            continue
        bin_index = int(np.searchsorted(bins, pt, side="right") - 1)
        if not (0 <= bin_index < nbins):
            continue
        for name in sumw:
            sumw[name][bin_index] += event.weights[name]
    for name in sumw:
        sumw[name] /= n_total
    return sumw


def fit_coefficients(sumw):
    X = np.array([[1.0, cHl3, cll1, cHl3 ** 2, cll1 ** 2, cHl3 * cll1] for _, cHl3, cll1 in BENCHMARKS])
    Y = np.array([sumw[name] for name, _, _ in BENCHMARKS])
    return np.linalg.solve(X, Y)


def predict(theta, cHl3, cll1):
    sm, lin_cHl3, lin_cll1, quad_cHl3, quad_cll1, mixed = theta
    return sm + lin_cHl3 * cHl3 + lin_cll1 * cll1 + quad_cHl3 * cHl3 ** 2 + quad_cll1 * cll1 ** 2 + mixed * cHl3 * cll1


def chi2(cHl3, cll1, channels):
    total = 0.0
    for theta, mu0, sigma2 in channels:
        mu = predict(theta, cHl3, cll1)
        total += np.sum((mu - mu0) ** 2 / sigma2)
    return total


def profile_chi2_over_cHl3(cll1_grid, cHl3_grid, channels):
    profiled = np.empty_like(cll1_grid)
    for i, cll1 in enumerate(cll1_grid):
        profiled[i] = min(chi2(cHl3, cll1, channels) for cHl3 in cHl3_grid)
    return profiled


def find_one_sigma_interval(x, chi2_values):
    i_min = np.argmin(chi2_values)
    target = chi2_values[i_min] + 1.0

    def crossing(indices):
        for a, b in zip(indices[:-1], indices[1:]):
            if (chi2_values[a] - target) * (chi2_values[b] - target) <= 0 and chi2_values[a] != chi2_values[b]:
                frac = (target - chi2_values[a]) / (chi2_values[b] - chi2_values[a])
                return x[a] + frac * (x[b] - x[a])
        return np.nan

    lo = crossing(range(i_min, -1, -1))
    hi = crossing(range(i_min, len(x)))
    return lo, hi


sumw = read_cross_sections(LHE_FILE, PT_BINS)
theta = fit_coefficients(sumw) * LUMI_FB * 1000.0

labels = ["SM", "cHl3 linear", "cll1 linear", "cHl3^2 quadratic", "cll1^2 quadratic", "cHl3*cll1 mixed"]
styles = ["-", "-", "--", "-", "--", "-"]

fig, ax = plt.subplots(figsize=(8, 5.5))
for coefficients, label, style in zip(theta, labels, styles):
    ax.stairs(np.abs(coefficients), PT_BINS, label=label, lw=2, ls=style)
ax.set_yscale("log")
ax.set_xlabel("leading lepton pT [GeV]")
ax.set_ylabel("|coefficient| [events]")
ax.set_title(f"WW: SM, linear, quadratic and mixed EFT distributions, L={LUMI_FB:.0f} fb-1")
ax.legend()
fig.tight_layout()
fig.savefig("distributions.png", dpi=150)

mu0 = theta[0]
sigma2 = np.maximum(mu0, 1.0)
channels = [(theta, mu0, sigma2)]

cll1_grid = np.linspace(-10, 10, 100)
cHl3_grid = np.linspace(-10, 10, 100)
chi2_profile = profile_chi2_over_cHl3(cll1_grid, cHl3_grid, channels)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(cll1_grid, chi2_profile, lw=2)
ax2.axhline(1.0, color="gray", ls=":")
ax2.set_xlabel("cll1")
ax2.set_ylabel("profiled chi2")
ax2.set_title("Profiled chi2 scan: cll1 (cHl3 profiled out)")
fig2.tight_layout()
fig2.savefig("chi2_profile_cll1.png", dpi=150)

lo, hi = find_one_sigma_interval(cll1_grid, chi2_profile)
print(f"{lo},{hi}")
