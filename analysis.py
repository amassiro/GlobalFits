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

Scale/PDF bands: every direct-generation series (SM, Lin, Quad, their sum
SM+Lin+Quad, and NP1's own nominal) gets a muR/muF scale envelope and an
NNPDF replica-based PDF band from MG5's `systematics` module (see
classify_systematics_weights/systematics_band below). SM+Lin+Quad's band is
formed by summing SM/Lin/Quad's per-variant histograms *before* reducing to
an envelope/sigma, so one shared muR/muF choice (or PDF replica) is varied
coherently across all three pieces of the decomposition, not as three
independent ones added in quadrature. create.sh forces use_syst=True for
all four MG5 outputs, including Lin/Quad, whose process definitions contain
a literal '^2' (NP^2==1/2) that makes MG5 auto-detect them as
"interference" and, by default, disable use_syst outright. MG5's own
source (madgraph/various/banner.py) and its own run_card.dat template both
warn, verbatim, that this systematics computation is wrong for
interference-type processes -- forced on anyway here (on request) to get a
band on every panel, but Lin/Quad's bands, and SM+Lin+Quad's (which sums
them), should be read as indicative, not as a validated uncertainty; the
plot labels them accordingly. The (rw) reweight-benchmark series (SM (rw)/
Lin (rw)/Quad (rw)) never get a band regardless of any of this: MG5's
systematics command only varies muR/muF/PDF around a sample's own nominal
weight, and reweight_card.dat's cHWB_0/p1/m1 points have no such variation
of their own.

Usage (normally via ./analysis.sh, which sets up the venv this needs):
    python analysis.py [--root DIR] [--bins N] [--mass-min GEV] [--mass-max GEV]
                        [--out FILE] [--yscale {linear,symlog}]
"""
from __future__ import annotations

import argparse
import re
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

# Shared caveat text for any Lin/Quad-derived series' plot annotation (module-
# level so compare_scales.py, which also builds SM+Lin+Quad via
# load_sm_lin_quad() below, can reuse the identical wording rather than
# re-deriving it -- see the module docstring's "Scale/PDF bands" paragraph).
INTERFERENCE_CAVEAT = ("scale/PDF band force-enabled for this NP^2\n"
                        "sample -- MG5's own code warns interference\n"
                        "systematics are wrong: indicative, not validated")


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


def classify_systematics_weights(header) -> tuple[list[str], list[str]]:
    """From a pylhe LHEHeader's <initrwgt> block, split the MG5 `systematics`
    weight entries (madgraph/various/systematics.py) into:

        scale_ids: the muR/muF 3x3-grid weight ids at the sample's own
                    dynamical scale choice (i.e. *no* DYN_SCALE attribute --
                    that attribute only marks the 4 *alternative fixed*
                    functional-form comparisons systematics.py also writes,
                    which are not part of the standard envelope), excluding
                    the (muR=1,muF=1) point (systematics.py never stores it
                    separately since it's identical to the event's own
                    nominal weight).
        pdf_ids:   the PDF-errorset member weight ids (muR=muF=1, no
                    DYN_SCALE), ordered by MemberID (member 0 first -- MG5
                    *does* store member 0 explicitly here, unlike the
                    (muR=1,muF=1) scale point, even though it is numerically
                    the same configuration as the nominal weight).

    Both lists are empty for a file with no <initrwgt> block at all, or an
    <initrwgt> that only has reweight_card-style weights (e.g. cHWB_0/p1/m1
    on PROC_cHWB_NP1: those weights carry no MUR/MUF attributes, so they
    never match either classification below).

    That used to also be the case for PROC_cHWB_linear/quadratic: MG5
    auto-detects their '^2' (NP^2==1/2) process definitions as
    "interference" and disables use_syst by default (see create.sh's
    dynamical_scale_choice comment for the sibling issue this same
    auto-detection causes). create.sh now forces use_syst back on for every
    sample regardless, so Lin/Quad do carry real scale/PDF weights here --
    but MG5's own source and its own run_card.dat template both warn,
    verbatim, that the systematics computation itself is wrong for
    interference-type processes. See the module docstring's "Scale/PDF
    bands" paragraph for how that caveat is surfaced downstream.
    """
    if header is None or not header.initrwgt.entries:
        return [], []

    scale_ids: list[str] = []
    pdf_entries: list[tuple[int, str]] = []
    for w in header.initrwgt.iter_weights():
        attrs = w.attributes
        if "ALPSFACT" in attrs or "MUR" not in attrs or "MUF" not in attrs:
            continue  # emission-scale variation, or a non-systematics weight (e.g. cHWB_0)
        is_dyn_alt = "DYN_SCALE" in attrs
        is_central_scale = attrs["MUR"] == "1.0" and attrs["MUF"] == "1.0"
        if is_central_scale and not is_dyn_alt:
            m = re.search(r"MemberID=(\d+)", w.name)
            pdf_entries.append((int(m.group(1)) if m else 0, w.id))
        elif not is_central_scale and not is_dyn_alt:
            scale_ids.append(w.id)

    pdf_entries.sort(key=lambda t: t[0])
    return scale_ids, [wid for _, wid in pdf_entries]


def nnpdf_replica_uncertainty(members: np.ndarray) -> np.ndarray:
    """Per-bin symmetric 1-sigma PDF uncertainty from N replica-set
    histograms (LHAPDF members 1..N -- member 0 / central excluded, shape
    (N, nbins)). This is the *exact* formula LHAPDF's own C++
    PDFSet::uncertainty() uses for ErrorType=='replicas' sets (its default,
    non-'alternative' branch: LHAPDF-6.5.5/src/PDFSet.cc lines 161-174,
    matching arXiv:1106.5788v2 Eqs. 2.3-2.4) -- the *unbiased* sample stddev
    of the replicas, i.e. N/(N-1) times the population variance. Verified
    against this LHAPDF source directly rather than assumed, since NNPDF's
    replica treatment is easy to get subtly wrong (e.g. using member 0 as
    the stats reference point instead of the replica average, or an N vs.
    N-1 denominator)."""
    n = members.shape[0]
    av = members.mean(axis=0)
    msq = (members ** 2).mean(axis=0)
    var = n / (n - 1.0) * (msq - av ** 2)
    return np.sqrt(np.clip(var, 0.0, None))


def systematics_weight_funcs(scale_ids: list[str], pdf_ids: list[str]) -> dict:
    """Build read_histograms()-compatible {name: function(event)->weight}
    entries for every scale/PDF systematics weight id, so they're
    accumulated in the *same* single pass over the file as the nominal
    weight (no second file read)."""
    funcs = {}
    for sid in scale_ids:
        funcs[f"scale#{sid}"] = lambda e, sid=sid: e.weights[sid]
    for pid in pdf_ids:
        funcs[f"pdf#{pid}"] = lambda e, pid=pid: e.weights[pid]
    return funcs


def systematics_band(sumw: dict, nominal: np.ndarray, scale_ids: list[str], pdf_ids: list[str]):
    """Turn the per-variant histograms accumulated via systematics_weight_funcs()
    into a plottable band dict, or None if this sample has no systematics
    weights at all (see classify_systematics_weights). Scale: envelope
    (min/max) over the 3x3 muR/muF grid *and* the nominal itself (9-point
    convention). PDF: nominal +/- the NNPDF replica sigma (nnpdf_replica_uncertainty),
    centered on the sample's actual nominal curve rather than the replica
    average -- standard practice, and justified here since NNPDF constructs
    member 0 to already closely track the replica average."""
    if not scale_ids and not pdf_ids:
        return None
    band = {}
    if scale_ids:
        variants = np.stack([sumw[f"scale#{sid}"] for sid in scale_ids] + [nominal])
        band["scale_lo"] = variants.min(axis=0)
        band["scale_hi"] = variants.max(axis=0)
    if pdf_ids:
        # pdf_ids[0] is MemberID=0 (the central member, redundant with `nominal`
        # up to MC noise) -- LHAPDF's own formula excludes it from the sigma sum.
        members = np.stack([sumw[f"pdf#{pid}"] for pid in pdf_ids[1:]])
        sigma = nnpdf_replica_uncertainty(members)
        band["pdf_lo"] = nominal - sigma
        band["pdf_hi"] = nominal + sigma
    return band


def open_lhe(path: Path):
    """Open an LHE(.gz) file and return the pylhe LHEFile object. pylhe reads
    the <header>/<init> blocks eagerly (before this call returns) and the
    <event> blocks lazily via a generator, so `.header` (needed to classify
    systematics weight ids -- classify_systematics_weights) is already
    available here, before any of `.events` has been consumed."""
    return pylhe.LHEFile.fromfile(str(path))


def read_histograms(path: Path, bins: np.ndarray, weight_funcs: dict, lhefile=None):
    """Read an LHE(.gz) file once. For each `name -> function(event)->weight`
    in weight_funcs, accumulate the weighted histogram (sumw) and its
    sum-of-weights-squared (sumw2, the MC statistical variance estimator).

    Combinations that mix several of an event's own weights (e.g. Lin (rw))
    must be done *inside* the function passed here, before this loop squares
    anything -- that's what makes the variance estimate correctly account
    for those weights' per-event correlation.

    `lhefile`, if given (an already-opened open_lhe(path) result), is used
    instead of re-opening/re-parsing `path` -- callers that need the header
    first (to build systematics_weight_funcs) already have it open.

    Returns (sumw: {name: array}, sumw2: {name: array}, n_total: int).
    """
    nbins = len(bins) - 1
    sumw = {k: np.zeros(nbins) for k in weight_funcs}
    sumw2 = {k: np.zeros(nbins) for k in weight_funcs}
    overflow = {k: 0 for k in weight_funcs}
    n_total = 0
    n_skipped = 0

    if lhefile is None:
        lhefile = open_lhe(path)
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


def load_sm_lin_quad(sm_file: Path, lin_file: Path, quad_file: Path, bins: np.ndarray) -> dict:
    """Histogram the SM/Lin/Quad direct-generation samples (each against its
    own nominal weight plus muR/muF/PDF systematics weights, one file read
    per sample) and combine them into the SM+Lin+Quad decomposition sum,
    including a coherently-combined scale/PDF band (see the module
    docstring's "Scale/PDF bands" paragraph for why the combination has to
    sum per-variant histograms *before* reducing to an envelope/sigma).

    Shared by analysis.py's own main() (panel 4: SM+Lin+Quad vs. NP1
    (nominal)) and compare_scales.py (which needs the identical SM+Lin+Quad
    baseline, generated at dynamical_scale_choice=-1, to compare against a
    separately-generated NP1 sample forced to dynamical_scale_choice=3).

    Returns a dict with keys:
        sm, sm_var, sm_band, n_sm
        lin, lin_var, lin_band, n_lin
        quad, quad_var, quad_band, n_quad
        slq, slq_var, slq_band            (the SM+Lin+Quad combination)
        scale_ids, pdf_ids                 (SM's -- == Lin's == Quad's, asserted below)
    """
    print("==> Histogramming direct samples (SM, Lin, Quad) ...")
    # All four MG5 outputs are generated with use_syst=True (create.sh forces
    # this even for Lin/Quad, whose process definitions contain the literal
    # '^2' that makes MG5 auto-detect them as "interference" and disable
    # use_syst by default -- see create.sh's comment above the sed loop that
    # overrides it back on, and classify_systematics_weights()'s docstring
    # for the caveat that comes with doing so). Fold each sample's own
    # muR/muF + PDF-errorset weights into the same single read pass rather
    # than re-reading the file.
    sm_lhe = open_lhe(sm_file)
    sm_scale_ids, sm_pdf_ids = classify_systematics_weights(sm_lhe.header)
    sm_funcs = {"SM": lambda e: e.eventinfo.weight}
    sm_funcs.update(systematics_weight_funcs(sm_scale_ids, sm_pdf_ids))
    sm_sumw, sm_sumw2, n_sm = read_histograms(sm_file, bins, sm_funcs, lhefile=sm_lhe)

    lin_lhe = open_lhe(lin_file)
    lin_scale_ids, lin_pdf_ids = classify_systematics_weights(lin_lhe.header)
    lin_funcs = {"Lin": lambda e: e.eventinfo.weight}
    lin_funcs.update(systematics_weight_funcs(lin_scale_ids, lin_pdf_ids))
    lin_sumw, lin_sumw2, n_lin = read_histograms(lin_file, bins, lin_funcs, lhefile=lin_lhe)

    quad_lhe = open_lhe(quad_file)
    quad_scale_ids, quad_pdf_ids = classify_systematics_weights(quad_lhe.header)
    quad_funcs = {"Quad": lambda e: e.eventinfo.weight}
    quad_funcs.update(systematics_weight_funcs(quad_scale_ids, quad_pdf_ids))
    quad_sumw, quad_sumw2, n_quad = read_histograms(quad_file, bins, quad_funcs, lhefile=quad_lhe)

    sm, sm_var = sm_sumw["SM"], sm_sumw2["SM"]
    lin, lin_var = lin_sumw["Lin"], lin_sumw2["Lin"]
    quad, quad_var = quad_sumw["Quad"], quad_sumw2["Quad"]
    sm_band = systematics_band(sm_sumw, sm, sm_scale_ids, sm_pdf_ids)
    lin_band = systematics_band(lin_sumw, lin, lin_scale_ids, lin_pdf_ids)
    quad_band = systematics_band(quad_sumw, quad, quad_scale_ids, quad_pdf_ids)
    for label, band in [("SM", sm_band), ("Lin", lin_band), ("Quad", quad_band)]:
        if band is None:
            print(f"    [info] {label}: no scale/PDF systematics weights found (was "
                  "./setup.sh's LHAPDF install run, and create.sh's systematics step, "
                  "completed before these events were generated?)")

    # SM+Lin+Quad's combined band must vary the *same* scale choice / PDF
    # replica coherently across all three samples before reducing to an
    # envelope/sigma -- summing each sample's own already-reduced band would
    # instead treat one shared muR/muF/PDF choice as three statistically
    # independent ones. That requires the three samples' weight ids to line
    # up 1-for-1 (same grid point / same replica member per id); assert it
    # rather than silently mis-combining if a future MG5/PDF-set change ever
    # breaks the alignment.
    if not (sm_scale_ids == lin_scale_ids == quad_scale_ids):
        sys.exit(
            "ERROR: SM/Lin/Quad scale-variation weight ids don't line up:\n"
            f"       SM={sm_scale_ids}\n       Lin={lin_scale_ids}\n       Quad={quad_scale_ids}\n"
            "       SM+Lin+Quad's combined scale band assumes the same muR/muF grid id-for-id."
        )
    if not (sm_pdf_ids == lin_pdf_ids == quad_pdf_ids):
        sys.exit(
            "ERROR: SM/Lin/Quad PDF-replica weight ids don't line up "
            f"(SM has {len(sm_pdf_ids)}, Lin has {len(lin_pdf_ids)}, Quad has {len(quad_pdf_ids)}).\n"
            "       SM+Lin+Quad's combined PDF band assumes the same replica id-for-id."
        )

    slq = sm + lin + quad
    slq_var = sm_var + lin_var + quad_var  # independent samples -> variances add
    slq_sumw = {k: sm_sumw[k] + lin_sumw[k] + quad_sumw[k] for k in sm_sumw if k != "SM"}
    slq_band = systematics_band(slq_sumw, slq, sm_scale_ids, sm_pdf_ids)

    return dict(
        sm=sm, sm_var=sm_var, sm_band=sm_band, n_sm=n_sm,
        lin=lin, lin_var=lin_var, lin_band=lin_band, n_lin=n_lin,
        quad=quad, quad_var=quad_var, quad_band=quad_band, n_quad=n_quad,
        slq=slq, slq_var=slq_var, slq_band=slq_band,
        scale_ids=sm_scale_ids, pdf_ids=sm_pdf_ids,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # ap.add_argument("--root", default=".", help="MASSIRO root dir (contains MG5_aMC_v2_9_27/)")
    ap.add_argument("--root", default=".", help="MASSIRO root dir (contains MG5_aMC_v3_7_2/)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--mass-min", type=float, default=20.0)
    ap.add_argument("--mass-max", type=float, default=500.0)
    ap.add_argument("--out", default="analysis_output/mll_comparison.png")
    ap.add_argument("--yscale", choices=["linear", "symlog"], default="linear",
                     help="Lin/Quad can be negative bin-by-bin (interference), "
                          "so log is not offered -- symlog handles sign.")
    args = ap.parse_args()



    # mg5_dir = Path(args.root) / "MG5_aMC_v2_9_27"
    mg5_dir = Path(args.root) / "MG5_aMC_v3_7_2"
    sm_file = require(mg5_dir / "PROC_SM_NP0" / "Events" / "run_01" / "unweighted_events.lhe.gz")
    lin_file = require(mg5_dir / "PROC_cHWB_linear" / "Events" / "run_01" / "unweighted_events.lhe.gz")
    quad_file = require(mg5_dir / "PROC_cHWB_quadratic" / "Events" / "run_01" / "unweighted_events.lhe.gz")
    np1_file = require(mg5_dir / "PROC_cHWB_NP1" / "Events" / "run_01" / "unweighted_events.lhe.gz")

    bins = np.linspace(args.mass_min, args.mass_max, args.bins + 1)

    slq_data = load_sm_lin_quad(sm_file, lin_file, quad_file, bins)
    sm, sm_var, sm_band, n_sm = (
        slq_data["sm"], slq_data["sm_var"], slq_data["sm_band"], slq_data["n_sm"])
    lin, lin_var, lin_band, n_lin = (
        slq_data["lin"], slq_data["lin_var"], slq_data["lin_band"], slq_data["n_lin"])
    quad, quad_var, quad_band, n_quad = (
        slq_data["quad"], slq_data["quad_var"], slq_data["quad_band"], slq_data["n_quad"])
    slq, slq_var, slq_band = slq_data["slq"], slq_data["slq_var"], slq_data["slq_band"]

    print("==> Histogramming PROC_cHWB_NP1 (nominal + reweight points cHWB_0/p1/m1) ...")
    # NP1's own nominal weight also has systematics weights (same reasoning as
    # SM above). The cHWB_0/p1/m1 *reweight_card* weights do not: MG5's
    # `systematics` command only varies muR/muF/PDF around a sample's own
    # nominal, so those benchmark points have no scale/PDF variation of their
    # own -- classify_systematics_weights() never matches them (no MUR/MUF
    # attributes on cHWB_* weights) so this is enforced structurally, not by
    # convention.
    np1_lhe = open_lhe(np1_file)
    np1_scale_ids, np1_pdf_ids = classify_systematics_weights(np1_lhe.header)
    np1_funcs = {
        "NP1": lambda e: e.eventinfo.weight,
        "SM (rw)": lambda e: e.weights["cHWB_0"],
        "Lin (rw)": lambda e: (e.weights["cHWB_p1"] - e.weights["cHWB_m1"]) / 2.0,
        "Quad (rw)": lambda e: (e.weights["cHWB_p1"] + e.weights["cHWB_m1"] - 2.0 * e.weights["cHWB_0"]) / 2.0,
    }
    np1_funcs.update(systematics_weight_funcs(np1_scale_ids, np1_pdf_ids))
    np1_sumw, np1_sumw2, n_np1 = read_histograms(np1_file, bins, np1_funcs, lhefile=np1_lhe)

    np1_nom, np1_nom_var = np1_sumw["NP1"], np1_sumw2["NP1"]
    sm_rw, sm_rw_var = np1_sumw["SM (rw)"], np1_sumw2["SM (rw)"]
    lin_rw, lin_rw_var = np1_sumw["Lin (rw)"], np1_sumw2["Lin (rw)"]
    quad_rw, quad_rw_var = np1_sumw["Quad (rw)"], np1_sumw2["Quad (rw)"]
    np1_band = systematics_band(np1_sumw, np1_nom, np1_scale_ids, np1_pdf_ids)

    bands_by_label = [("SM", sm, sm_band), ("Lin", lin, lin_band), ("Quad", quad, quad_band),
                       ("SM+Lin+Quad", slq, slq_band), ("NP1 (nominal)", np1_nom, np1_band)]
    if any(band is not None for _, _, band in bands_by_label):
        print("\n==> Scale/PDF variation, integrated over the plotted range "
              "(cross-check against the numbers `./bin/madevent systematics` itself printed;\n"
              "    Lin/Quad/SM+Lin+Quad are force-enabled and NOT validated -- see module "
              "docstring):")
        for label, nominal, band in bands_by_label:
            if band is None:
                continue
            tot = nominal.sum()
            parts = []
            if "scale_lo" in band:
                lo, hi = band["scale_lo"].sum(), band["scale_hi"].sum()
                # NB: for a negative `tot` (Lin can be, being pure SM x EFT
                # interference) lo/hi's printed signs appear "swapped" from
                # the usual positive-cross-section convention -- e.g. the
                # numerically *smaller* (more negative) envelope edge prints
                # as the *positive*-percent entry. Both numbers are still
                # exactly 100*(edge/tot - 1); nothing here special-cases the
                # sign, it's just what that formula gives for a negative tot.
                parts.append(f"scale [{100*(lo/tot-1):+.1f}%, {100*(hi/tot-1):+.1f}%]")
            if "pdf_lo" in band:
                lo, hi = band["pdf_lo"].sum(), band["pdf_hi"].sum()
                # abs(): pdf_hi = nominal + sigma, pdf_lo = nominal - sigma
                # with sigma >= 0 always, so this is symmetric by construction
                # (100*(hi/tot-1) == -100*(lo/tot-1) exactly, for any sign of
                # tot) -- it's a magnitude, hence the literal "+/-" prefix.
                # Without abs(), a negative `tot` (Lin can be one) flips the
                # computed number's own sign and prints a confusing "+/--N%".
                parts.append(f"PDF +/-{abs(100*(hi/tot-1)):.2f}%")
            print(f"    {label:<16s} {tot: .6g}   " + "   ".join(parts))

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

    no_syst_rw = ("no scale/PDF systematics:\n(rw) has no variation of its own --\n"
                  "MG5's systematics only varies muR/muF/PDF\naround a sample's own nominal weight")
    comparisons = [
        dict(name_a="SM", a=sm, a_var=sm_var, band_a=sm_band,
             name_b="SM (rw)", b=sm_rw, b_var=sm_rw_var, band_b=None, note_b=no_syst_rw,
             color="tab:blue"),
        dict(name_a="Lin", a=lin, a_var=lin_var, band_a=lin_band, note_a=INTERFERENCE_CAVEAT,
             name_b="Lin (rw)", b=lin_rw, b_var=lin_rw_var, band_b=None, note_b=no_syst_rw,
             color="tab:orange"),
        dict(name_a="Quad", a=quad, a_var=quad_var, band_a=quad_band, note_a=INTERFERENCE_CAVEAT,
             name_b="Quad (rw)", b=quad_rw, b_var=quad_rw_var, band_b=None, note_b=no_syst_rw,
             color="tab:green"),
        dict(name_a="SM+Lin+Quad", a=slq, a_var=slq_var, band_a=slq_band, note_a=INTERFERENCE_CAVEAT,
             name_b="NP1 (nominal)", b=np1_nom, b_var=np1_nom_var, band_b=np1_band,
             color="black"),
    ]
    make_plot(bins, comparisons, args.out, args.yscale)


def make_plot(bins: np.ndarray, comparisons: list, out_path_str: str, yscale: str,
              suptitle: str = r"$pp \to \ell^+\ell^-$: cHWB SM/Lin/Quad -- direct generation vs. reweighting"):
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
        squeeze=False,  # keep axes a true 2D (2, ncols) array even when ncols==1
        # (matplotlib otherwise squeezes any size-1 dimension away, and
        # compare_scales.py legitimately calls this with a single comparison
        # -- axes[0, col]/axes[1, col] below need the 2D shape unconditionally)
    )

    for col, cmp in enumerate(comparisons):
        ax, rax = axes[0, col], axes[1, col]
        name_a, a, a_var, band_a = cmp["name_a"], cmp["a"], cmp["a_var"], cmp.get("band_a")
        name_b, b, b_var, band_b = cmp["name_b"], cmp["b"], cmp["b_var"], cmp.get("band_b")
        color = cmp["color"]

        # Scale (envelope, lighter/wider) and PDF (NNPDF replica sigma, darker/
        # narrower) bands -- only ever set for the *direct*-generation series
        # (SM, Lin, Quad, SM+Lin+Quad, NP1 (nominal)), never for the (rw)
        # reweight-benchmark series: MG5's systematics command only varies
        # muR/muF/PDF around a sample's own nominal weight, and
        # reweight_card.dat's cHWB_0/p1/m1 points have no such variation of
        # their own (see classify_systematics_weights()'s docstring). The
        # Lin/Quad/SM+Lin+Quad bands additionally carry a note_a caveat
        # (set in main()) since MG5 itself disclaims this computation for
        # interference-type processes.
        for band, label in [(band_a, name_a), (band_b, name_b)]:
            if band is None:
                continue
            # step="mid": draw each bin's [lo, hi] as a flat rectangle spanning
            # the bin (transitioning halfway to the next center), not a point
            # linearly interpolated between bin centers -- with these samples'
            # very uneven per-bin statistics (e.g. the near-threshold peak
            # falling off by orders of magnitude within a couple of bins), a
            # plain linear fill_between draws misleading diagonal wedges
            # between distant bin values instead of each bin's own band.
            if "scale_lo" in band:
                ax.fill_between(centers, band["scale_lo"], band["scale_hi"], step="mid",
                                 color=color, alpha=0.15, linewidth=0,
                                 label=f"{label} scale env.")
            if "pdf_lo" in band:
                ax.fill_between(centers, band["pdf_lo"], band["pdf_hi"], step="mid",
                                 color=color, alpha=0.30, linewidth=0,
                                 label=f"{label} PDF unc.")

        ax.errorbar(centers - dodge, a, yerr=np.sqrt(a_var), fmt="o", ms=3, capsize=2,
                    lw=1, color=color, label=name_a)
        ax.errorbar(centers + dodge, b, yerr=np.sqrt(b_var), fmt="s", ms=3, capsize=2,
                    lw=1, color=color, markerfacecolor="none", label=name_b)
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.set_yscale(yscale)
        ax.set_title(f"{name_a}  vs  {name_b}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        if col == 0:
            ax.set_ylabel("events / bin (weighted)")

        note = cmp.get("note") or "\n".join(filter(None, [cmp.get("note_a"), cmp.get("note_b")]))
        if note:
            ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=6, color="dimgrey",
                    style="italic", ha="right", va="top",
                    bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))

        ratio, ratio_unc = ratio_with_unc(a, a_var, b, b_var)
        rax.errorbar(centers, ratio, yerr=ratio_unc, fmt=".", ms=4, capsize=2, lw=1, color=color)
        rax.axhline(1.0, color="grey", lw=0.5)
        rax.set_ylim(0.0, 2.0)
        rax.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
        if col == 0:
            rax.set_ylabel("ratio (B / A)")

    # wrap=True: analysis.py's own call site is a wide 4-panel figure where its
    # (short) default suptitle always fit on one line, but compare_scales.py
    # legitimately calls this with a single, narrower panel and a longer,
    # comparison-specific title -- without wrapping it just overflows past the
    # figure edge instead of reflowing to a second line.
    fig.suptitle(suptitle, wrap=True)
    fig.savefig(out_path, dpi=150)
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
