#!/usr/bin/env python3
"""
analysis.py -- compare the dilepton invariant mass (m_ll) distribution across
the cHWB SMEFT samples produced by create.sh, direct generation vs.
reweighting, with MC statistical uncertainties, as four separate panels:

    (1) SM             vs  SM (rw)         PROC_SM_NP0               vs  PROC_cHWB_NP1 @ cHWB_0
    (2) Lin            vs  Lin (rw)        PROC_cHWB_linear          vs  PROC_cHWB_NP1 @ (cHWB_p1-cHWB_m1)/2
    (3) Quad           vs  Quad (rw)       PROC_cHWB_quadratic       vs  PROC_cHWB_NP1 @ (cHWB_p1+cHWB_m1-2*cHWB_0)/2
    (4) SM+Lin+Quad    vs  NP1 (nominal)   (1)+(2)+(3) summed        vs  PROC_cHWB_NP1's own un-reweighted weight

Why these four: create.sh's RESTRICT card fixes cHWB to a generic
non-special value (~1, not exactly 1) while leaving it externally settable,
so sigma(c) = sigma_SM + c*sigma_lin + c^2*sigma_quad. reweight_card.dat on
PROC_cHWB_NP1 evaluates that same sigma(c) at c = 0, +1, -1 (weights
"cHWB_0"/"cHWB_p1"/"cHWB_m1"); the finite differences of those three isolate
sigma_SM, sigma_lin, sigma_quad the same way SM/Lin/Quad were generated
directly, and PROC_cHWB_NP1's own nominal (un-reweighted) weight is exactly
sigma_SM+sigma_lin+sigma_quad at that same benchmark. Panels 1-3 cross-check
the decomposition against the reweighting; panel 4 cross-checks the
decomposition against the un-reweighted NP1 generation directly (no
reweighting involved at all).

Uncertainties: each bin carries the standard MC statistical variance
estimator sum(w_i^2) over the events landing in that bin (ROOT's Sumw2
convention -- appropriate here since LHE events are unweighted-per-sample
but not literally unit weight, and Lin/Quad weights can be signed). For
Lin (rw)/Quad (rw), the combination is formed *per event* from that event's
correlated cHWB_0/p1/m1 weights before squaring/binning, which correctly
propagates their correlation (they come from the same NP1 events). SM+Lin+
Quad's variance is the quadrature sum of the three independent samples'
variances. Every ratio (rw / direct, and NP1-nominal / SM+Lin+Quad) compares
two *independent* MG5 runs, so ratio uncertainties use ordinary uncorrelated
error propagation.

Usage (normally via ./analysis.sh, which sets up the venv this needs):
    python analysis.py [--root DIR] [--bins N] [--mass-min GEV] [--mass-max GEV]
                        [--out FILE] [--yscale {linear,symlog}]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pylhe
except ImportError:
    sys.exit(
        "ERROR: pylhe is not installed in this Python environment.\n"
        "       Run ./analysis.sh (it creates a venv and installs pylhe) instead\n"
        "       of invoking analysis.py directly."
    )

LEPTON_PDG_IDS = {11, 13, 15}  # e, mu, tau -- charge sign doesn't matter, |id| is checked


def dilepton_mass(event) -> float | None:
    """Invariant mass of the two final-state charged leptons in an LHE event.
    Returns None if the event doesn't have exactly 2 final-state leptons
    (should not happen for p p > l+ l-, but skip defensively rather than crash)."""
    leptons = [p for p in event.particles if p.status == 1 and abs(p.id) in LEPTON_PDG_IDS]
    if len(leptons) != 2:
        return None
    e = leptons[0].e + leptons[1].e
    px = leptons[0].px + leptons[1].px
    py = leptons[0].py + leptons[1].py
    pz = leptons[0].pz + leptons[1].pz
    m2 = e * e - px * px - py * py - pz * pz
    return float(np.sqrt(m2)) if m2 > 0 else 0.0


def read_histograms(path: Path, bins: np.ndarray, weight_funcs: dict):
    """Read an LHE(.gz) file once. For each `name -> function(event)->weight`
    in weight_funcs, accumulate the weighted histogram (sumw) and its
    sum-of-weights-squared (sumw2, the MC statistical variance estimator).

    Combinations that mix several of an event's own weights (e.g. Lin (rw))
    must be done *inside* the function passed here, before this loop squares
    anything -- that's what makes the variance estimate correctly account
    for those weights' per-event correlation.

    Returns (sumw: {name: array}, sumw2: {name: array}, n_total: int).
    """
    nbins = len(bins) - 1
    sumw = {k: np.zeros(nbins) for k in weight_funcs}
    sumw2 = {k: np.zeros(nbins) for k in weight_funcs}
    overflow = {k: 0 for k in weight_funcs}
    n_total = 0
    n_skipped = 0

    lhefile = pylhe.LHEFile.fromfile(str(path))
    for event in lhefile.events:
        n_total += 1
        mass = dilepton_mass(event)
        if mass is None:
            n_skipped += 1
            continue

        idx = int(np.searchsorted(bins, mass, side="right") - 1)
        in_range = 0 <= idx < nbins

        for name, func in weight_funcs.items():
            try:
                w = func(event)
            except KeyError as exc:
                sys.exit(
                    f"ERROR: {path} has no {exc} weight in its <rwgt> blocks.\n"
                    f"       Available: {sorted(event.weights)}.\n"
                    f"       Did create.sh's reweight step actually run for this sample?"
                )
            if in_range:
                sumw[name][idx] += w
                sumw2[name][idx] += w * w
            else:
                overflow[name] += 1

    if n_skipped:
        print(f"    [warn] {path.name}: skipped {n_skipped}/{n_total} events "
              f"without exactly 2 final-state charged leptons")
    for name, n_over in overflow.items():
        if n_over:
            print(f"    [warn] {path.name} [{name}]: {n_over}/{n_total} events "
                  f"fell outside [{bins[0]:.1f}, {bins[-1]:.1f}] GeV (widen --mass-max?)")

    return sumw, sumw2, n_total


def require(path: Path) -> Path:
    if not path.is_file():
        sys.exit(
            f"ERROR: missing {path}\n"
            f"       Run ./create.sh first to generate (and reweight) events for all four samples."
        )
    return path


def ratio_with_unc(a: np.ndarray, a_var: np.ndarray, b: np.ndarray, b_var: np.ndarray):
    """ratio = b/a and its uncertainty, for two statistically *independent*
    quantities (true for every comparison made here -- each pair always
    comes from two different MG5 runs). Uses absolute (not relative)
    variance propagation so it stays well-defined when b == 0 in a bin;
    only blows up (-> NaN) where the direct-sample bin a is itself empty."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(a) > 0, b / a, np.nan)
        var_ratio = np.where(np.abs(a) > 0, (b ** 2 * a_var + a ** 2 * b_var) / a ** 4, np.nan)
        unc = np.sqrt(var_ratio)
    return ratio, unc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="MASSIRO root dir (contains MG5_aMC_v2_9_27/)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--mass-min", type=float, default=20.0)
    ap.add_argument("--mass-max", type=float, default=500.0)
    ap.add_argument("--out", default="analysis_output/mll_comparison.png")
    ap.add_argument("--yscale", choices=["linear", "symlog"], default="linear",
                     help="Lin/Quad can be negative bin-by-bin (interference), "
                          "so log is not offered -- symlog handles sign.")
    args = ap.parse_args()

    mg5_dir = Path(args.root) / "MG5_aMC_v2_9_27"
    sm_file = require(mg5_dir / "PROC_SM_NP0" / "Events" / "run_01" / "unweighted_events.lhe.gz")
    lin_file = require(mg5_dir / "PROC_cHWB_linear" / "Events" / "run_01" / "unweighted_events.lhe.gz")
    quad_file = require(mg5_dir / "PROC_cHWB_quadratic" / "Events" / "run_01" / "unweighted_events.lhe.gz")
    np1_file = require(mg5_dir / "PROC_cHWB_NP1" / "Events" / "run_01" / "unweighted_events.lhe.gz")

    bins = np.linspace(args.mass_min, args.mass_max, args.bins + 1)

    print("==> Histogramming direct samples (SM, Lin, Quad) ...")
    sm_sumw, sm_sumw2, n_sm = read_histograms(sm_file, bins, {"SM": lambda e: e.eventinfo.weight})
    lin_sumw, lin_sumw2, n_lin = read_histograms(lin_file, bins, {"Lin": lambda e: e.eventinfo.weight})
    quad_sumw, quad_sumw2, n_quad = read_histograms(quad_file, bins, {"Quad": lambda e: e.eventinfo.weight})

    sm, sm_var = sm_sumw["SM"], sm_sumw2["SM"]
    lin, lin_var = lin_sumw["Lin"], lin_sumw2["Lin"]
    quad, quad_var = quad_sumw["Quad"], quad_sumw2["Quad"]

    slq = sm + lin + quad
    slq_var = sm_var + lin_var + quad_var  # independent samples -> variances add

    print("==> Histogramming PROC_cHWB_NP1 (nominal + reweight points cHWB_0/p1/m1) ...")
    np1_funcs = {
        "NP1": lambda e: e.eventinfo.weight,
        "SM (rw)": lambda e: e.weights["cHWB_0"],
        "Lin (rw)": lambda e: (e.weights["cHWB_p1"] - e.weights["cHWB_m1"]) / 2.0,
        "Quad (rw)": lambda e: (e.weights["cHWB_p1"] + e.weights["cHWB_m1"] - 2.0 * e.weights["cHWB_0"]) / 2.0,
    }
    np1_sumw, np1_sumw2, n_np1 = read_histograms(np1_file, bins, np1_funcs)

    np1_nom, np1_nom_var = np1_sumw["NP1"], np1_sumw2["NP1"]
    sm_rw, sm_rw_var = np1_sumw["SM (rw)"], np1_sumw2["SM (rw)"]
    lin_rw, lin_rw_var = np1_sumw["Lin (rw)"], np1_sumw2["Lin (rw)"]
    quad_rw, quad_rw_var = np1_sumw["Quad (rw)"], np1_sumw2["Quad (rw)"]

    table = [
        ("SM", sm, sm_var),
        ("Lin", lin, lin_var),
        ("Quad", quad, quad_var),
        ("SM+Lin+Quad", slq, slq_var),
        ("NP1 (nominal)", np1_nom, np1_nom_var),
        ("SM (rw)", sm_rw, sm_rw_var),
        ("Lin (rw)", lin_rw, lin_rw_var),
        ("Quad (rw)", quad_rw, quad_rw_var),
    ]
    print("\n==> Total yield per category (sum of weights +/- MC stat. unc., over the plotted range):")
    for name, h, v in table:
        print(f"    {name:<16s} {h.sum(): .6g}  +/-  {np.sqrt(v.sum()):.6g}")
    print(f"\n==> events read: SM={n_sm}  Lin={n_lin}  Quad={n_quad}  NP1={n_np1}")

    comparisons = [
        ("SM", sm, sm_var, "SM (rw)", sm_rw, sm_rw_var, "tab:blue"),
        ("Lin", lin, lin_var, "Lin (rw)", lin_rw, lin_rw_var, "tab:orange"),
        ("Quad", quad, quad_var, "Quad (rw)", quad_rw, quad_rw_var, "tab:green"),
        ("SM+Lin+Quad", slq, slq_var, "NP1 (nominal)", np1_nom, np1_nom_var, "black"),
    ]
    make_plot(bins, comparisons, args.out, args.yscale)


def make_plot(bins: np.ndarray, comparisons: list, out_path_str: str, yscale: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    centers = 0.5 * (bins[:-1] + bins[1:])
    dodge = 0.12 * (bins[1:] - bins[:-1])  # small x-offset so the two error bars don't hide each other

    ncols = len(comparisons)
    fig, axes = plt.subplots(
        2, ncols, figsize=(4.6 * ncols, 6.5), sharex="col",
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08),
        constrained_layout=True,
    )

    for col, (name_a, a, a_var, name_b, b, b_var, color) in enumerate(comparisons):
        ax, rax = axes[0, col], axes[1, col]

        ax.errorbar(centers - dodge, a, yerr=np.sqrt(a_var), fmt="o", ms=3, capsize=2,
                    lw=1, color=color, label=name_a)
        ax.errorbar(centers + dodge, b, yerr=np.sqrt(b_var), fmt="s", ms=3, capsize=2,
                    lw=1, color=color, markerfacecolor="none", label=name_b)
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.set_yscale(yscale)
        ax.set_title(f"{name_a}  vs  {name_b}", fontsize=10)
        ax.legend(fontsize=8)
        if col == 0:
            ax.set_ylabel("events / bin (weighted)")

        ratio, ratio_unc = ratio_with_unc(a, a_var, b, b_var)
        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt=".", ms=4, capsize=2, lw=1, color=color)
        rax.axhline(1.0, color="grey", lw=0.5)
        rax.set_ylim(0.0, 2.0)
        rax.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
        if col == 0:
            rax.set_ylabel("ratio (B / A)")

    fig.suptitle(r"$pp \to \ell^+\ell^-$: cHWB SM/Lin/Quad -- direct generation vs. reweighting")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
