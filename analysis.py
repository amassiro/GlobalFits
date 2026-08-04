#!/usr/bin/env python3
"""
analysis.py -- compare the dilepton invariant mass (m_ll) distribution across
the cHWB SMEFT samples produced by create.sh:

    SM             PROC_SM_NP0          (NP=0, pure Standard Model)
    Lin            PROC_cHWB_linear     (NP=1 NP^2==1, SM x EFT interference)
    Quad           PROC_cHWB_quadratic  (NP=1 NP^2==2, pure EFT^2)
    SM+Lin+Quad    the above three summed bin-by-bin
    SM (rw)        PROC_cHWB_NP1 reweighted to cHWB=0        (weight "cHWB_0")
    Lin (rw)       PROC_cHWB_NP1 reweight combination        (cHWB_p1 - cHWB_m1) / 2
    Quad (rw)      PROC_cHWB_NP1 reweight combination        (cHWB_p1 + cHWB_m1 - 2*cHWB_0) / 2

Why the reweight combination: create.sh's RESTRICT card fixes cHWB to a
generic non-special value (~1, not exactly 1) while leaving it externally
settable, so sigma(c) = sigma_SM + c*sigma_lin + c^2*sigma_quad. The
reweight_card.dat on PROC_cHWB_NP1 evaluates that same sigma(c) at c = 0, +1,
-1 (weights "cHWB_0"/"cHWB_p1"/"cHWB_m1"), and those three finite differences
isolate sigma_SM, sigma_lin, sigma_quad exactly the way SM/Lin/Quad were
generated directly. Comparing the two is a cross-check of both methods; "SM
(rw)" should track "SM", etc., and "SM+Lin+Quad" should track PROC_cHWB_NP1's
own nominal (un-reweighted) weight, since all four samples sit at the same
cHWB benchmark.

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


def read_histograms(path: Path, bins: np.ndarray, weight_keys: list):
    """Read an LHE(.gz) file once and return {key: weighted_histogram} for
    each entry in weight_keys, plus the total event count.

    A key of None uses each event's nominal weight (event.eventinfo.weight).
    A string key uses event.weights[key] -- an MG5 reweight point named via
    reweight_card.dat's --rwgt_name=<key>.
    """
    nbins = len(bins) - 1
    counts = {k: np.zeros(nbins) for k in weight_keys}
    overflow = {k: 0 for k in weight_keys}
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

        for key in weight_keys:
            if key is None:
                w = event.eventinfo.weight
            else:
                try:
                    w = event.weights[key]
                except KeyError:
                    sys.exit(
                        f"ERROR: {path} has no '{key}' weight in its <rwgt> blocks.\n"
                        f"       Available: {sorted(event.weights)}.\n"
                        f"       Did create.sh's reweight step actually run for this sample?"
                    )
            if in_range:
                counts[key][idx] += w
            else:
                overflow[key] += 1

    if n_skipped:
        print(f"    [warn] {path.name}: skipped {n_skipped}/{n_total} events "
              f"without exactly 2 final-state charged leptons")
    for key, n_over in overflow.items():
        if n_over:
            label = key if key else "nominal"
            print(f"    [warn] {path.name} [{label}]: {n_over}/{n_total} events "
                  f"fell outside [{bins[0]:.1f}, {bins[-1]:.1f}] GeV (widen --mass-max?)")

    return counts, n_total


def require(path: Path) -> Path:
    if not path.is_file():
        sys.exit(
            f"ERROR: missing {path}\n"
            f"       Run ./create.sh first to generate (and reweight) events for all four samples."
        )
    return path


def safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denom) > 0, numer / denom, np.nan)


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
    sm_counts, n_sm = read_histograms(sm_file, bins, [None])
    lin_counts, n_lin = read_histograms(lin_file, bins, [None])
    quad_counts, n_quad = read_histograms(quad_file, bins, [None])
    sm, lin, quad = sm_counts[None], lin_counts[None], quad_counts[None]
    sm_lin_quad = sm + lin + quad

    print("==> Histogramming PROC_cHWB_NP1 reweight points (cHWB_0/p1/m1) ...")
    np1_counts, n_np1 = read_histograms(np1_file, bins, ["cHWB_0", "cHWB_p1", "cHWB_m1"])
    rw0, rwp1, rwm1 = np1_counts["cHWB_0"], np1_counts["cHWB_p1"], np1_counts["cHWB_m1"]

    sm_rw = rw0
    lin_rw = (rwp1 - rwm1) / 2.0
    quad_rw = (rwp1 + rwm1 - 2.0 * rw0) / 2.0

    series = {
        "SM": sm,
        "Lin": lin,
        "Quad": quad,
        "SM+Lin+Quad": sm_lin_quad,
        "SM (rw)": sm_rw,
        "Lin (rw)": lin_rw,
        "Quad (rw)": quad_rw,
    }

    print("\n==> Total yield per category (sum of weights over the plotted range):")
    for name, h in series.items():
        print(f"    {name:<14s} {h.sum(): .6g}")
    print(f"\n==> events read: SM={n_sm}  Lin={n_lin}  Quad={n_quad}  NP1={n_np1}")

    make_plot(bins, series, args.out, args.yscale)


def make_plot(bins: np.ndarray, series: dict, out_path_str: str, yscale: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = {
        "SM":          dict(color="tab:blue",   ls="-"),
        "Lin":         dict(color="tab:orange", ls="-"),
        "Quad":        dict(color="tab:green",  ls="-"),
        "SM+Lin+Quad": dict(color="black",      ls="-", lw=2),
        "SM (rw)":     dict(color="tab:blue",   ls="--"),
        "Lin (rw)":    dict(color="tab:orange", ls="--"),
        "Quad (rw)":   dict(color="tab:green",  ls="--"),
    }

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08),
    )

    for name, h in series.items():
        ax.stairs(h, bins, label=name, **styles[name])
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.set_ylabel("events / bin (weighted)")
    ax.set_yscale(yscale)
    ax.set_title(r"$pp \to \ell^+\ell^-$: cHWB SM/Lin/Quad, direct vs. reweighted")
    ax.legend(fontsize=9, ncol=2)

    # ratio panel: reweighted / direct, for the three pairs that have both
    for name in ("SM", "Lin", "Quad"):
        ratio = safe_ratio(series[f"{name} (rw)"], series[name])
        centers = 0.5 * (bins[:-1] + bins[1:])
        rax.plot(centers, ratio, drawstyle="steps-mid", color=styles[name]["color"], label=name)
    rax.axhline(1.0, color="grey", lw=0.5)
    rax.set_ylim(0.5, 1.5)
    rax.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
    rax.set_ylabel("rw / direct")

    fig.savefig(out_path, dpi=150)
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
