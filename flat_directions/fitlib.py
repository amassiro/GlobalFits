#!/usr/bin/env python3
"""
fitlib.py -- shared LHE-reading, exact-quadratic-response-surface-fitting,
and Fisher/PCA/Minuit helpers for the "flat directions" notebook.

Used *identically* for both samples in the lecture:
    - PROC_WW_emu_NP1  (create_ww.sh)    p p > e+ ve mu- vm~ (+c.c.), NP=1
    - PROC_CCDY_e_NP1  (create_ccdy.sh)  p p > e+ ve (+c.c.), NP=1
That's deliberate, not laziness: steps 1-5 of the lecture are the *same*
code path called twice on two different LHE files, which is exactly what
makes step 9 (compare the two Fisher matrices) an apples-to-apples
comparison instead of "two different analyses that happen to both produce
a matrix".

Everything here treats reweight_points.py's OPERATORS list and its exact
quadratic benchmark grid (fit_points()/validation_points()) as the single
source of truth -- this module never hardcodes an operator name or a
benchmark point.

Pipeline (see load_sample() for the one-call version):
    1. read_yields()            LHE(.gz) -> per-benchmark, per-bin (sumw, sumw2)
    2. fit_quadratic_surface()  28 benchmark yields -> exact theta(bin) such
                                 that sigma_bin(c) = theta_row(c) . theta[:,bin]
    3. validate_surface()       6 held-out points -> predicted vs. MG5-actual
    4. scale_theta_to_counts()  pb -> expected event counts at an assumed
                                 luminosity (Asimov dataset lives here)
    5. linear_fisher_matrix()   the textbook EFT Fisher information matrix,
                                 built from theta's *linear* coefficients
    6. eigh_sorted()            PCA: diagonalize the Fisher matrix
    7. fit_full() / fit_reduced()   iminuit MIGRAD/HESSE/MINOS, k=0 (SM) start
    8. chi2_curve_1d() / profile_1d() / sigma_from_curve()
                                 naive ("fix everything else to 0") vs.
                                 genuine profile-likelihood 1D scans -- see
                                 the notebook's step 10 competition section
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pylhe
except ImportError:
    sys.exit(
        "ERROR: pylhe not installed in this Python environment.\n"
        "       Use ../.analysis_venv (see ../analysis.sh) or `pip install pylhe`."
    )

from iminuit import Minuit

import reweight_points as rwp

# --------------------------------------------------------------------------
# 1. LHE reading
# --------------------------------------------------------------------------

CHARGED_LEPTON_IDS = {11, 13, 15}  # e, mu, tau (|id| -- charge doesn't matter)


def open_lhe(path):
    """See ../analysis.py's open_lhe(): pylhe reads <header>/<init> eagerly
    and <event> blocks lazily, so this is cheap even before .events is
    iterated."""
    return pylhe.LHEFile.fromfile(str(path))


def leading_lepton_pt(event) -> float | None:
    """Leading final-state charged-lepton p_T [GeV]. The primary
    differential observable for *both* samples here: WW has two charged
    leptons (e, mu) in the final state, CC-DY has exactly one (e) -- "leading"
    degrades gracefully to "the one there is". TGC-type and current-type
    dimension-6 operators alike grow the amplitude with energy, so the
    high-p_T tail carries most of the discrimination power between
    benchmark points (same reasoning real ATLAS/CMS aTGC/SMEFT analyses
    use for this kind of final state)."""
    pts = [
        float(np.hypot(p.px, p.py))
        for p in event.particles
        if p.status == 1 and abs(p.id) in CHARGED_LEPTON_IDS
    ]
    return max(pts) if pts else None


def read_yields(path, bins, weight_names, observable=dijet_mass, selection=None,
                 lhefile=None):
    """One pass over an LHE(.gz) file: for every benchmark name in
    weight_names (must match a `launch --rwgt_name=<name>` entry in the
    reweight_card.dat this sample was generated with -- see
    reweight_points.reweight_card_text()), accumulate a binned (sumw,
    sumw2) histogram of `observable`.

    `selection`, if given, is a callable(event) -> bool (e.g.
    vbs_selection()) applied BEFORE `observable`; events failing it are
    dropped entirely (not counted in n_skipped/overflow, and contribute no
    weight to any benchmark) -- so passing selection=vbs_selection makes
    every returned quantity describe that fiducial region exactly, the
    same way as if a differently-cut LHE file had been read.

    Returns (sumw: {name: array[nbins]}, sumw2: {name: array[nbins]}, n_total: int).
    sumw is in physical cross-section units (pb).

    IMPORTANT normalization: MadGraph's unweighted_events.lhe(.gz) writes
    XWGTUP = the sample's TOTAL cross section (identical to the <init>
    block's XSECUP) into every single event, not xsec/n_total -- verified
    directly against this exercise's own LHE file (<init> XSECUP matches
    the first several events' XWGTUP bit-for-bit). Summing that raw weight
    over n_total events therefore overshoots the true cross section by a
    factor of n_total; the correct per-event contribution is
    event.weights[name] / n_total, so the two accumulators below are
    divided by n_total (sumw2 by n_total**2, since it's a sum of squared
    per-event weights) right before returning. This matches the convention
    used in LLR_MCinHEP/HEP_differential_xsec.ipynb (`xsec =
    sum(event.eventinfo.weight)/nevents`). Applying the division here
    (once, after the accumulation loop) rather than per-event inside the
    loop is numerically equivalent (n_total is a fixed constant) and
    avoids a second pass over the file.
    """
    nbins = len(bins) - 1
    sumw = {name: np.zeros(nbins) for name in weight_names}
    sumw2 = {name: np.zeros(nbins) for name in weight_names}
    n_total = 0
    n_selected = 0
    n_skipped = 0
    overflow = 0

    if lhefile is None:
        lhefile = open_lhe(path)
    for event in lhefile.events:
        n_total += 1
        if selection is not None and not selection(event):
            continue
        n_selected += 1
        obs = observable(event)
        if obs is None:
            n_skipped += 1
            continue
        idx = int(np.searchsorted(bins, obs, side="right") - 1)
        if not (0 <= idx < nbins):
            overflow += 1
            continue
        for name in weight_names:
            try:
                w = event.weights[name]
            except KeyError as exc:
                sys.exit(
                    f"ERROR: {path} has no {exc} weight in its <rwgt> blocks.\n"
                    f"       Available: {sorted(event.weights)}.\n"
                    f"       Did the reweight step (madevent reweight run_01 -from_cards)\n"
                    f"       actually complete for this sample?"
                )
            sumw[name][idx] += w
            sumw2[name][idx] += w * w

    if selection is not None:
        frac = n_selected / n_total if n_total else 0.0
        print(f"    [selection] {Path(path).name}: {n_selected}/{n_total} events "
              f"passed ({frac:.1%})")
    if n_skipped:
        denom = n_selected if selection is not None else n_total
        print(f"    [warn] {Path(path).name}: skipped {n_skipped}/{denom} events "
              f"with fewer than 2 final-state jets")
    if overflow:
        denom = n_selected if selection is not None else n_total
        print(f"    [warn] {Path(path).name}: {overflow}/{denom} events fell "
              f"outside [{bins[0]:.1f}, {bins[-1]:.1f}] GeV (widen `bins`?)")

    for name in weight_names:
        sumw[name] /= n_total
        sumw2[name] /= n_total ** 2

    return sumw, sumw2, n_total


# --------------------------------------------------------------------------
# 2. Exact quadratic response-surface fit
#
#    sigma_bin(c) = theta_0 + sum_i theta_i c_i + sum_i theta_ii c_i^2
#                   + sum_{i<j} theta_ij c_i c_j
#
#    reweight_points.fit_points() supplies exactly one benchmark per
#    unknown (1 + N + N + C(N,2) points for N operators), so this is an
#    *exactly determined* linear system, not a least-squares fit -- see
#    that module's docstring for the finite-difference logic that makes it
#    solvable by construction (SM point fixes theta_0; the +-1 pair per
#    operator separates theta_i (odd in c_i) from theta_ii (even in c_i);
#    each (+1,+1) pair point then fixes exactly one cross term theta_ij,
#    since everything else it depends on is already known).
# --------------------------------------------------------------------------

def term_labels(operators=rwp.OPERATORS) -> list[str]:
    """Human-readable label for each of the 1+N+N+C(N,2) design-matrix
    columns, in the exact order design_row() emits them."""
    labels = ["1"] + list(operators) + [f"{op}^2" for op in operators]
    for i, oi in enumerate(operators):
        for oj in operators[i + 1:]:
            labels.append(f"{oi}*{oj}")
    return labels


def design_row(cvals: dict, operators=rwp.OPERATORS) -> np.ndarray:
    c = np.array([cvals.get(op, 0.0) for op in operators])
    cross = [c[i] * c[j] for i in range(len(operators)) for j in range(i + 1, len(operators))]
    return np.concatenate([[1.0], c, c ** 2, cross])


def design_matrix(points, operators=rwp.OPERATORS) -> np.ndarray:
    return np.array([design_row(cvals, operators) for _, cvals in points])


def fit_quadratic_surface(yields: dict, points=None, operators=rwp.OPERATORS) -> np.ndarray:
    """yields: {name: array[nbins]} (e.g. the `sumw` from read_yields()).
    points defaults to reweight_points.fit_points() (the 28-point exact
    grid; NOT the validation points -- those are held out on purpose, see
    validate_surface()).

    Returns theta, shape (n_terms, n_bins) -- one column per bin, one row
    per term (term_labels() gives the row order)."""
    if points is None:
        points = rwp.fit_points()
    names = [name for name, _ in points]
    X = design_matrix(points, operators)               # (n_terms, n_terms)
    Y = np.array([yields[name] for name in names])      # (n_terms, n_bins)
    return np.linalg.solve(X, Y)                         # (n_terms, n_bins)


def predict(theta: np.ndarray, cvals: dict, operators=rwp.OPERATORS) -> np.ndarray:
    """theta as returned by fit_quadratic_surface() (or scale_theta_to_counts()
    below) -- evaluate the fitted surface at an arbitrary Wilson-coefficient
    point. predict(theta, {}) (i.e. c=0) always returns exactly theta[0, :],
    the SM/Asimov prediction."""
    return design_row(cvals, operators) @ theta


def validate_surface(theta_pb: np.ndarray, yields: dict, points=None,
                      operators=rwp.OPERATORS) -> list[dict]:
    """For each held-out point (default: reweight_points.validation_points(),
    NOT used to fit theta_pb) compare the surface's prediction to MG5's own
    reweighted yield. Returns one summary dict per point -- run this BEFORE
    trusting anything built on top of the surface (Fisher matrix, Minuit
    fits, ...); large max_abs_rel_diff means either too few MC events
    (statistical noise in the 28 fit-point yields feeds directly into
    theta) or a bug upstream."""
    if points is None:
        points = rwp.validation_points()
    rows = []
    for name, cvals in points:
        pred = predict(theta_pb, cvals, operators)
        actual = yields[name]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(actual != 0, (pred - actual) / actual, 0.0)
        rows.append(dict(
            name=name, cvals=cvals,
            pred_total=float(pred.sum()), actual_total=float(actual.sum()),
            max_abs_rel_diff=float(np.max(np.abs(rel))),
        ))
    return rows

def theta_linear_only(theta_counts: np.ndarray, operators=rwp.OPERATORS) -> np.ndarray:
    """Copy of theta_counts with every quadratic/cross-term row zeroed out
    -- i.e. the SM+linear-only truncation of the exact quadratic surface.
    predict(theta_linear_only(theta), cvals, operators) then gives the
    "assume the EFT response is exactly linear in c" prediction, using the
    exact same predict()/design_row() machinery as everywhere else
    (design_row() still computes the c^2/cross columns, they just get
    multiplied by zero here). Used throughout step 4/5 to compare "with"
    vs "without" the quadratic term without ever touching the underlying
    MG5 yields -- both fits read the same 21-point-exact surface, just
    with different rows switched off."""
    n_ops = len(operators)
    theta_lin = theta_counts.copy()
    theta_lin[1 + n_ops:, :] = 0.0
    return theta_lin

# --------------------------------------------------------------------------
# 3. Luminosity scaling -- pb surface -> expected-event-count surface
# --------------------------------------------------------------------------

def scale_theta_to_counts(theta_pb: np.ndarray, lumi_fb: float = 300.0) -> np.ndarray:
    """sigma[pb] -> N[expected events] = sigma * L is linear in sigma, so
    the *entire* quadratic surface can be rescaled by this one constant
    (1 pb = 1000 fb, hence the 1000 factor to go from fb^-1 to pb^-1).
    Default 300 fb^-1: a round, standard "Run 3-ish / early HL-LHC"
    single-experiment benchmark for a hands-on exercise -- change it and
    re-run to see the Fisher matrix (and hence which directions count as
    "flat") depend on assumed statistics, exactly as it should."""
    return theta_pb * (lumi_fb * 1000.0)


def asimov_variance(mu0_counts: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Poisson variance of the Asimov dataset (data == c=0 expectation
    exactly, by construction -- see chi2_factory). `floor` keeps
    near-empty high-p_T tail bins from causing a divide-by-zero; you
    wouldn't trust sensitivity claimed from a <1-expected-event bin
    anyway."""
    return np.maximum(mu0_counts, floor)


# --------------------------------------------------------------------------
# 4. Fisher matrix / PCA
# --------------------------------------------------------------------------

def linear_fisher_matrix(theta_counts: np.ndarray, sigma2: np.ndarray,
                          operators=rwp.OPERATORS) -> np.ndarray:
    """F_ij = sum_bins (dN_bin/dc_i)(dN_bin/dc_j) / sigma_bin^2, evaluated
    at c=0 -- the standard linearized EFT Fisher information matrix (e.g.
    Hessian/2 of the Gaussian chi2 at its minimum). Built directly from
    theta_counts' *linear* rows (index 1..n_ops; row 0 is the intercept,
    quadratic/cross rows are not needed for this local, linear-order
    object) -- pure linear algebra, always well-defined even where the full
    nonlinear Minuit fit below might struggle numerically."""
    n_ops = len(operators)
    A = theta_counts[1:1 + n_ops, :]     # (n_ops, n_bins) = dN/dc_i per bin
    return (A / sigma2) @ A.T             # (n_ops, n_ops)


def eigh_sorted(F: np.ndarray):
    """Eigen-decomposition of a symmetric matrix, sorted by DESCENDING
    eigenvalue (index 0 = best-constrained direction, index -1 = flattest).
    Returns (eigvals[n], eigvecs[n,n]) with eigvecs[:, k] the k-th
    eigenvector (unit-normalized, numpy convention)."""
    w, v = np.linalg.eigh(F)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def correlation_matrix(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def chi2_scan_2d(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                  op_x: str, op_y: str, grid: np.ndarray,
                  operators=rwp.OPERATORS, fixed: dict | None = None):
    """Evaluate the TRUE nonlinear chi2(c) (same object chi2_factory() builds
    for Minuit) on a 2D (op_x, op_y) grid, holding every other operator
    fixed (default: fixed at 0, i.e. the SM point) -- a visualisable
    'conditional slice' through the full n_ops-dimensional likelihood.
    This is deliberately the SAME predict()/theta_counts surface used
    everywhere else in this module, called at O(grid^2) points instead of
    fit by MIGRAD -- useful to *see* a flat valley (step 6) or to overlay
    two processes' valleys to compare their orientation directly (step 7/9),
    rather than only reading the orientation off an eigenvector.

    Returns (X, Y, Z) meshgrid arrays ready for plt.contour/contourf; Z is
    chi2, so the SM point (c=0) sits at Z=0 by construction (Asimov data).
    """
    fixed = {} if fixed is None else fixed
    X, Y = np.meshgrid(grid, grid)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cvals = dict(fixed)
            cvals[op_x] = X[i, j]
            cvals[op_y] = Y[i, j]
            mu = predict(theta_counts, cvals, operators)
            Z[i, j] = np.sum((mu - mu0) ** 2 / sigma2)
    return X, Y, Z


def chi2_curve_1d(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                   op: str, grid: np.ndarray, fixed: dict | None = None,
                   operators=rwp.OPERATORS) -> np.ndarray:
    """1D sibling of chi2_scan_2d(): scan a single operator over `grid`,
    holding every other operator at exactly 0 except whatever is explicitly
    named in `fixed` (which can pin them to a non-zero value too, though the
    step 10 competition section below only ever uses 0). This is the
    "naive" slice through the likelihood -- nothing but `op` is allowed to
    move -- exactly the (usually optimistic, and outright wrong whenever a
    real correlation is present) "fit to X only" analysis, contrasted
    against the genuine profile likelihood in profile_1d() below."""
    fixed = {} if fixed is None else fixed
    chi2 = np.empty_like(grid, dtype=float)
    for k, v in enumerate(grid):
        cvals = dict(fixed)
        cvals[op] = v
        mu = predict(theta_counts, cvals, operators)
        chi2[k] = np.sum((mu - mu0) ** 2 / sigma2)
    return chi2


def profile_1d(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                op: str, grid: np.ndarray, profile_ops: list[str],
                fixed: dict | None = None, operators=rwp.OPERATORS):
    """The genuine PROFILE likelihood: scan `op` over `grid`, and at every
    single point let every operator in `profile_ops` re-minimize chi2
    (everything else -- not `op`, not in `profile_ops`, not in `fixed` --
    stays at 0). This is exactly what MINOS does internally to get an
    asymmetric confidence interval on one parameter of a multi-parameter
    fit, generalised here into an explicit, plottable curve. Contrast with
    chi2_curve_1d(), which never lets anything but `op` move -- that
    contrast (naive vs. profiled) is the whole point of step 10.

    Returns (chi2: array[len(grid)], best_fit: {profile_op: array[len(grid)]});
    the second return value lets you see e.g. how far cHl3 has to wander to
    keep compensating for each trial value of cll1."""
    fixed = {} if fixed is None else fixed
    chi2 = np.empty_like(grid, dtype=float)
    best_fit = {p: np.empty_like(grid, dtype=float) for p in profile_ops}
    for k, v in enumerate(grid):
        def _chi2(p, _v=v):
            cvals = dict(fixed)
            cvals[op] = _v
            cvals.update(zip(profile_ops, p))
            mu = predict(theta_counts, cvals, operators)
            return float(np.sum((mu - mu0) ** 2 / sigma2))

        m = Minuit(_chi2, np.zeros(len(profile_ops)), name=list(profile_ops))
        m.errordef = Minuit.LEAST_SQUARES
        for name in profile_ops:
            m.limits[name] = (-20.0, 20.0)
        m.migrad()
        chi2[k] = m.fval
        for p in profile_ops:
            best_fit[p][k] = m.values[p]
    return chi2, best_fit


def sigma_from_curve(grid: np.ndarray, chi2vals: np.ndarray, delta: float = 1.0) -> float:
    """Read a 1-sigma uncertainty directly off a 1D chi2(c) curve: walk
    outward from the curve's own minimum in both directions until it
    crosses chi2_min + delta (delta=1 <-> the standard 68% CL interval for
    one parameter), linearly interpolating between grid points, and return
    half the total width. Used throughout step 10 to turn the naive/
    profiled curves from chi2_curve_1d()/profile_1d() into single headline
    sigma(cll1) numbers -- cross-checked there against Minuit's own HESSE
    uncertainty from the equivalent fixed/free fit."""
    kmin = int(np.argmin(chi2vals))
    target = chi2vals[kmin] + delta

    def _cross(idx_range):
        for a, b in zip(idx_range[:-1], idx_range[1:]):
            if (chi2vals[a] - target) * (chi2vals[b] - target) <= 0 and chi2vals[a] != chi2vals[b]:
                frac = (target - chi2vals[a]) / (chi2vals[b] - chi2vals[a])
                return grid[a] + frac * (grid[b] - grid[a])
        return np.nan

    lo = _cross(range(kmin, -1, -1))
    hi = _cross(range(kmin, len(grid)))
    if np.isnan(lo) or np.isnan(hi):
        return float("nan")
    return 0.5 * (hi - lo)


def as_frame(matrix: np.ndarray, operators=rwp.OPERATORS, index=None, columns=None):
    """Label a square (or rectangular) matrix with operator names for
    display in a notebook. Falls back to bare numpy printing if pandas
    isn't available (it is in this repo's venv, but this keeps the rest of
    fitlib usable without it)."""
    try:
        import pandas as pd
    except ImportError:
        return matrix
    idx = index if index is not None else list(operators)
    col = columns if columns is not None else list(operators)
    return pd.DataFrame(matrix, index=idx, columns=col)


# --------------------------------------------------------------------------
# 5. Minuit likelihood -- Asimov data (== SM, c=0), fit initialized at 0
# --------------------------------------------------------------------------

def chi2_factory(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                  operators=rwp.OPERATORS):
    """Build a vector-call chi2(c) for iminuit. This is the TRUE
    nonlinear-in-c object from the full quadratic (SM+interference+EFT^2)
    surface -- squaring (mu(c)-mu0) turns a quadratic-in-c model into a
    quartic-in-c chi2, so a Minuit fit of it is sensitive to genuine flat
    *valleys*, not just the flat tangent-plane directions linear_fisher_matrix()
    sees. mu0 IS the c=0 (SM) prediction exactly (Asimov technique: "data"
    has no statistical fluctuation), so c=0 is guaranteed to be a global
    minimum with chi2=0 -- what MIGRAD/HESSE/MINOS actually have to tell
    you is how *tightly curved* that minimum is in each direction, i.e.
    exactly the flat-direction question."""
    def chi2(c):
        cvals = dict(zip(operators, c))
        mu = predict(theta_counts, cvals, operators)
        return float(np.sum((mu - mu0) ** 2 / sigma2))
    return chi2


def _run_migrad_hesse_minos(m: Minuit) -> Minuit:
    m.migrad()
    m.hesse()
    try:
        m.minos()
    except Exception as exc:
        print(f"    [minos] raised {type(exc).__name__}: {exc}\n"
              f"    (expected if a direction is exactly/near flat -- MINOS "
              f"has to find where chi2 rises by 1, and can't if it never "
              f"does within the scan range. Check m.merrors[...].is_valid "
              f"and the HESSE covariance/correlation matrix instead.)")
    return m


def fit_full(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
             operators=rwp.OPERATORS, start=None, limit: float | None = 5.0,
             fixed: list[str] | None = None) -> Minuit:
    """Fit all len(operators) Wilson coefficients simultaneously, MIGRAD-
    initialized at k=0 (SM) as requested. `limit` (default +-5, a few times
    the +-1 benchmark range the surface was fit/validated on) keeps MIGRAD
    from wandering into a region where the quadratic-in-c truncation is no
    longer a trustworthy model of the underlying physics -- MINOS errors
    hitting that boundary is itself a valid (and common, in real fits)
    symptom of a flat direction, not a bug to hide. `fixed`, if given, pins
    the named operators to their start value (0) instead of leaving them
    free -- used by the step 10 competition section to restrict a fit to
    just (cHl3, cll1) without needing a whole new quadratic surface (fixing
    an operator to exactly 0 in this exact quadratic surface is
    mathematically identical to never having included it -- every term
    that touches it just evaluates to 0)."""
    chi2 = chi2_factory(theta_counts, mu0, sigma2, operators)
    start = np.zeros(len(operators)) if start is None else np.asarray(start)
    m = Minuit(chi2, start, name=list(operators))
    m.errordef = Minuit.LEAST_SQUARES
    if limit is not None:
        for name in operators:
            m.limits[name] = (-limit, limit)
    if fixed:
        for name in fixed:
            m.fixed[name] = True
    return _run_migrad_hesse_minos(m)


def fit_reduced(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                 V: np.ndarray, k_keep: int, operators=rwp.OPERATORS,
                 start=None, limit: float | None = 20.0) -> Minuit:
    """PCA-reduced fit: only the k_keep best-constrained eigendirections of
    V (columns sorted by DESCENDING eigenvalue -- see eigh_sorted()) are
    left free; the remaining flat/near-flat directions are fixed at
    exactly 0 by construction (c = V[:, :k_keep] @ d never populates them).
    This calls the *same* predict() surface as fit_full(), just
    reparametrised -- so "the reduced fit converges" is a statement about
    the same physical model, not a different, easier one."""
    Vk = V[:, :k_keep]

    def chi2(d):
        c = Vk @ d
        cvals = dict(zip(operators, c))
        mu = predict(theta_counts, cvals, operators)
        return float(np.sum((mu - mu0) ** 2 / sigma2))

    start = np.zeros(k_keep) if start is None else np.asarray(start)
    names = [f"d{k + 1}" for k in range(k_keep)]
    m = Minuit(chi2, start, name=names)
    m.errordef = Minuit.LEAST_SQUARES
    if limit is not None:
        for name in names:
            m.limits[name] = (-limit, limit)
    return _run_migrad_hesse_minos(m)


# --------------------------------------------------------------------------
# 6. One-call pipeline
# --------------------------------------------------------------------------

def load_sample(lhe_path, bins: np.ndarray, lumi_fb: float = 300.0,
                 observable=leading_lepton_pt, operators=rwp.OPERATORS) -> dict:
    """Read one LHE(.gz) sample and run steps 1-4 above in one call. Returns
    a dict with everything steps 5+ (Fisher/PCA/Minuit, done separately so
    you can inspect/plot intermediate results) need:

        bins, sumw, sumw2, n_total          -- raw histograms (pb)
        theta_pb                            -- quadratic surface, pb
        validation                          -- validate_surface() rows
        theta                               -- quadratic surface, expected counts @ lumi_fb
        mu0, sigma2                         -- Asimov data & variance (counts)
    """
    fit_pts = rwp.fit_points()
    val_pts = rwp.validation_points()
    all_names = [n for n, _ in fit_pts] + [n for n, _ in val_pts]

    sumw, sumw2, n_total = read_yields(lhe_path, bins, all_names, observable=observable)
    print(f"    {Path(lhe_path).name}: {n_total} events read, "
          f"SM yield in range = {sumw['SM'].sum():.4g} pb")

    theta_pb = fit_quadratic_surface(sumw, fit_pts, operators)
    validation = validate_surface(theta_pb, sumw, val_pts, operators)
    worst = max(validation, key=lambda r: r["max_abs_rel_diff"])
    print(f"    surface validation: worst held-out point '{worst['name']}', "
          f"max |Delta bin| / bin = {worst['max_abs_rel_diff']:.2%}")

    theta = scale_theta_to_counts(theta_pb, lumi_fb)
    mu0 = theta[0, :]
    sigma2 = asimov_variance(mu0)

    return dict(
        bins=bins, sumw=sumw, sumw2=sumw2, n_total=n_total,
        theta_pb=theta_pb, validation=validation,
        theta=theta, mu0=mu0, sigma2=sigma2,
        lumi_fb=lumi_fb, operators=list(operators),
    )
