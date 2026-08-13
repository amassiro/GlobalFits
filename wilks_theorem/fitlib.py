#!/usr/bin/env python3
"""
fitlib.py -- shared LHE-reading, exact-quadratic-response-surface-fitting,
Fisher/PCA/Minuit, and toy-coverage helpers for the "Wilks' theorem" VBS
hands-on exercise.

Direct sibling of ../flat_directions/fitlib.py -- sections 1-5 below are
copied essentially verbatim (same design: reweight_points.OPERATORS and its
exact quadratic benchmark grid are the single source of truth; this module
never hardcodes an operator name or a benchmark point). What's new here,
specific to this exercise, is:

    - dijet_mass()                the m_jj observable (VBS-tagging-jet pair);
                                   superseded as the primary observable by
                                   mt_ww() (step 3), still used for the
                                   VBS-tagging selection cut
    - ww_mass()                   truth-level m_WW observable (2 leptons + 2
                                   neutrinos), NOT a realistic analysis
                                   variable (unreconstructable in real data)
                                   -- used only as a diagnostic proving
                                   dijet_mass()'s EFT-growth flatness is a
                                   m_jj-vs-m_WW decorrelation effect, not a
                                   parametrization bug (step 3 Aside)
    - mt_ww()                     dilepton+MET transverse mass, the
                                   detector-reconstructable proxy for
                                   ww_mass() -- this exercise's primary
                                   observable from step 3 on (replaces
                                   dijet_mass() in that role)
    - theta_linear_only()         SM+linear-only truncation of the exact
                                   quadratic surface, for "with vs without
                                   the quadratic term" fits (step 4)
    - linear_vs_quadratic_shape_ratio() a per-operator scalar diagnostic,
                                   computed from theta_pb with NO statistical
                                   weighting, used to sort operators into
                                   linear-dominated / comparable / quadratic-
                                   dominated (step 5) -- luminosity-INDEPENDENT
                                   by construction, see its docstring
    - linear_vs_quadratic_ratio() the Fisher-weighted analogue: same idea,
                                   evaluated at each operator's own 1-sigma
                                   point, so it IS luminosity-dependent --
                                   answers a different question (step 5/6)
    - lnN_vector() + weighted_dot() a customizable, fully-correlated
                                   log-normal-style normalization systematic
                                   (fixed at the nominal/SM yield, profiled
                                   in closed form via a Sherman-Morrison
                                   rank-1 update -- see weighted_dot()'s
                                   docstring): tuning its size moves an
                                   operator from linear- to quadratic-
                                   dominated at FIXED luminosity, the lnN
                                   analogue of what scanning lumi_fb itself
                                   would do (step 3) -- off by default
                                   (lnn_v=None), which reproduces every
                                   function below bit-for-bit as if this
                                   feature didn't exist
    - coverage_scan() (+ helpers) toy-MC Delta-chi^2=1 coverage vs. the
                                   68.27% Wilks' theorem target (step 6),
                                   following arXiv:2207.01350's own Gaussian-
                                   fixed-covariance toy methodology, now with
                                   an optional two-part (correlated lnN pull
                                   + independent per-bin) toy noise model

Pipeline (see load_sample() for the one-call version):
    1. read_yields()              LHE(.gz) -> per-benchmark, per-bin (sumw, sumw2)
    2. fit_quadratic_surface()    21 benchmark yields -> exact theta(bin) such
                                   that sigma_bin(c) = theta_row(c) . theta[:,bin]
    3. validate_surface()         6 held-out points -> predicted vs. MG5-actual
    4. scale_theta_to_counts()    pb -> expected event counts at an assumed
                                   luminosity (Asimov dataset lives here)
    5. lnN_vector()  (optional)   build the rank-1 correlated-systematic
                                   vector at a chosen size; pass as lnn_v=
                                   to every step below to switch it on
    6. linear_vs_quadratic_shape_ratio() (classification) and
                                   linear_fisher_matrix() / quadratic_fisher_diag() /
                                   linear_vs_quadratic_ratio() (Fisher-weighted cross-check)
    7. chi2_curve_1d() (full theta and theta_linear_only) + sigma_from_curve()
                                   "with vs without the quadratic term" (step 4/5)
    8. coverage_scan()            toy Delta-chi^2=1 coverage vs. Wilks (step 6)
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

CHARGED_LEPTON_IDS = {11, 13, 15}   # e, mu, tau (|id| -- charge doesn't matter)
JET_IDS = {1, 2, 3, 4, 5, 21}       # d, u, s, c, b, g (massless restriction)
NEUTRINO_IDS = {12, 14, 16}         # ve, vm, vt -- the only invisible final-
                                     # state particles at this LHE (parton,
                                     # no-shower) level, so MET is exactly
                                     # their vector-sum p_T, see missing_et()


def open_lhe(path):
    """See ../flat_directions/fitlib.py's open_lhe(): pylhe reads
    <header>/<init> eagerly and <event> blocks lazily, so this is cheap
    even before .events is iterated."""
    return pylhe.LHEFile.fromfile(str(path))


def _final_state(event, pdg_ids: set) -> list:
    """Status==1 (final-state) particles whose |PDG id| is in `pdg_ids`."""
    return [p for p in event.particles if p.status == 1 and abs(p.id) in pdg_ids]


def _leading_two_by_pt(particles: list):
    """The 2 leading-p_T objects from `particles`, p_T-descending, or None
    if fewer than 2 are present. Shared by every dijet/dilepton observable
    AND cut below so mjj/deltaEtajj (and mll/ptl1/ptl2) always agree on
    which 2 objects they mean -- selecting jets independently for the mass
    cut and the Delta-eta cut would risk silently picking different pairs."""
    if len(particles) < 2:
        return None
    particles = sorted(particles, key=lambda p: np.hypot(p.px, p.py), reverse=True)
    return particles[0], particles[1]


def _invariant_mass(*particles) -> float:
    """Invariant mass of the sum of ANY number of 4-vectors -- not just
    pairs. Existing 2-body call sites (dilepton_mass, dijet_mass) already
    call this as `_invariant_mass(*pair)`, which still unpacks to exactly
    2 positional args, so this generalization is call-site-compatible.
    ww_mass() below is the reason for the generalization: it needs the
    invariant mass of a 4-particle (2 lepton + 2 neutrino) system."""
    e = sum(p.e for p in particles)
    px = sum(p.px for p in particles)
    py = sum(p.py for p in particles)
    pz = sum(p.pz for p in particles)
    m2 = e * e - px * px - py * py - pz * pz
    return float(np.sqrt(max(m2, 0.0)))


def _eta(p) -> float | None:
    """Pseudorapidity eta = asinh(pz/pT). None on the measure-zero p_T==0
    edge case (rather than raising), consistent with this module's
    None-means-'can't compute this observable for this event' convention."""
    pt = float(np.hypot(p.px, p.py))
    if pt <= 0.0:
        return None
    return float(np.arcsinh(p.pz / pt))


def leading_lepton_pt(event) -> float | None:
    """Leading final-state charged-lepton p_T [GeV]. Not this exercise's
    primary observable (that's dijet_mass() below), but kept available as
    a bonus/comparison variable -- this same-sign WW-VBS final state also
    has 2 charged leptons, exactly like the flat-directions WW sample."""
    pts = [float(np.hypot(p.px, p.py)) for p in _final_state(event, CHARGED_LEPTON_IDS)]
    return max(pts) if pts else None


def lepton_pts(event) -> tuple[float, float] | None:
    """(pT of leading lepton, pT of subleading lepton) [GeV], or None if
    fewer than 2 final-state charged leptons -- feeds the ptl1/ptl2 VBS
    selection cuts below."""
    pair = _leading_two_by_pt(_final_state(event, CHARGED_LEPTON_IDS))
    if pair is None:
        return None
    l1, l2 = pair
    return float(np.hypot(l1.px, l1.py)), float(np.hypot(l2.px, l2.py))


def dilepton_mass(event) -> float | None:
    """Invariant mass [GeV] of the 2 leading-p_T final-state charged
    leptons -- the mll VBS selection cut below."""
    pair = _leading_two_by_pt(_final_state(event, CHARGED_LEPTON_IDS))
    if pair is None:
        return None
    return _invariant_mass(*pair)


def missing_et(event) -> float:
    """Missing transverse energy [GeV] = |vector-sum p_T| of the 2
    final-state neutrinos. At this LHE (parton, no-shower, no-detector)
    level the neutrinos are the only invisible particles in this final
    state, so this is exact truth-level MET, not a visible-recoil proxy --
    equivalently the negative vector-sum p_T of every OTHER final-state
    particle, since the 2 incoming partons have zero net p_T. Always
    returns a float (0.0, not None, on the also-measure-zero <2-neutrino
    edge case) since MET>min is a lower-bound cut, not a "must exist"
    selection."""
    nus = _final_state(event, NEUTRINO_IDS)
    px = sum(p.px for p in nus)
    py = sum(p.py for p in nus)
    return float(np.hypot(px, py))


def dijet_mass(event) -> float | None:
    """Invariant mass of the two leading-p_T light-parton ('jet') final-
    state particles [GeV] -- the classic VBS-tagging-jet-pair observable,
    and this exercise's primary discriminating variable (step 3).

    At this fixed-order QCD=0 LHE level there are exactly 2 such partons
    per event (no extra QCD radiation was generated -- this is parton-
    level, no shower), but we still explicitly rank/select the leading 2
    by p_T rather than assuming order or count, so this degrades
    gracefully/robustly rather than silently mis-pairing if that ever
    isn't true (e.g. if this is ever pointed at a sample with real
    radiation)."""
    pair = _leading_two_by_pt(_final_state(event, JET_IDS))
    if pair is None:
        return None
    return _invariant_mass(*pair)


def dijet_deltaeta(event) -> float | None:
    """|Delta eta| between the two leading-p_T VBS-tagging jets -- the
    classic forward/backward-tagging-jet rapidity-gap VBS selection cut.
    Uses the SAME leading-2-jets pair as dijet_mass() (see
    _leading_two_by_pt())."""
    pair = _leading_two_by_pt(_final_state(event, JET_IDS))
    if pair is None:
        return None
    j1, j2 = pair
    eta1, eta2 = _eta(j1), _eta(j2)
    if eta1 is None or eta2 is None:
        return None
    return float(abs(eta1 - eta2))


def ww_mass(event) -> float | None:
    """Invariant mass [GeV] of the full leptonic system (the 2 final-state
    charged leptons + 2 neutrinos) -- i.e. of the W+W+ pair at parton
    level, equivalently sqrt(shat) of the VV scattering subsystem the EFT
    operators in this exercise actually couple to. This is a genuinely
    different quantity from dijet_mass(): dijet_mass measures the
    *tagging-quark* kinematics, only weakly correlated event-by-event with
    the energy flowing through the VV vertex in an inclusive, fixed-order
    (no parton shower) 2->6 sample like this one -- see the step-3 Aside.

    NOT usable as a real analysis observable: with 2 escaping neutrinos the
    leptonic system is kinematically underconstrained in actual detector
    data (unlike e.g. single-neutrino W->lv, there's no missing-pz
    solution), so this exists here purely as a truth-level diagnostic
    variable -- mt_ww() below is the reconstructable proxy actually used
    as this exercise's primary observable from step 3 on."""
    leps = _final_state(event, CHARGED_LEPTON_IDS)
    nus = _final_state(event, NEUTRINO_IDS)
    if len(leps) < 2 or len(nus) < 2:
        return None
    return _invariant_mass(*leps, *nus)


def mt_ww(event) -> float | None:
    """Transverse mass [GeV] of the dilepton + missing-E_T system -- the
    detector-level-reconstructable proxy for ww_mass() (the full W+W+
    invariant mass), built only from the 2 leading charged leptons'
    transverse momenta/energy and the (in real data, measured) missing
    transverse momentum -- no dependence on the neutrinos'
    individually-unmeasurable longitudinal momenta, unlike ww_mass().

    Standard "dilepton+MET" transverse-mass definition used e.g. in
    ATLAS/CMS WW/VBS cross-section measurements:

        E_T^ll = sqrt(m_ll^2 + (p_T^ll)^2)
        m_T^2  = (E_T^ll + MET)^2 - |p_T^ll + p_T^miss|^2

    Reduces to the familiar single-W m_T formula in the collinear limit
    where p_T^miss is aligned/anti-aligned with p_T^ll. Uses the same
    leading-2-lepton pair as dilepton_mass()/lepton_pts() (see
    _leading_two_by_pt()) and the same neutrino vector-sum as
    missing_et(), so this stays internally consistent with both.

    None if fewer than 2 final-state charged leptons (same not-enough-
    objects convention as the other observables above); unlike missing_et(),
    a "too few neutrinos" event still gets a well-defined answer (MET=0),
    since m_T only needs the *momentum*, not a lepton-neutrino pairing."""
    pair = _leading_two_by_pt(_final_state(event, CHARGED_LEPTON_IDS))
    if pair is None:
        return None
    l1, l2 = pair
    px_ll, py_ll = l1.px + l2.px, l1.py + l2.py
    pt_ll = float(np.hypot(px_ll, py_ll))
    et_ll = float(np.hypot(_invariant_mass(l1, l2), pt_ll))

    nus = _final_state(event, NEUTRINO_IDS)
    px_miss = sum(p.px for p in nus)
    py_miss = sum(p.py for p in nus)
    met = float(np.hypot(px_miss, py_miss))

    ex, ey = px_ll + px_miss, py_ll + py_miss
    mt2 = (et_ll + met) ** 2 - (ex * ex + ey * ey)
    return float(np.sqrt(max(mt2, 0.0)))


def vbs_selection(event, mjj_min: float = 500.0, mll_min: float = 50.0,
                   ptl_min: float = 30.0, met_min: float = 20.0,
                   deltaeta_jj_min: float = 2.5) -> bool:
    """Standard VBS-tagging fiducial selection: mjj > mjj_min, mll >
    mll_min, both lepton pT > ptl_min, MET > met_min, and
    |Delta eta_jj| > deltaeta_jj_min, all simultaneously. Added while
    investigating why the fully-inclusive cW_p1/SM benchmark ratio
    (step 3) looked flat vs. m_jj instead of rising steeply as expected
    for a dimension-6, energy-growing operator.

    That flatness is NOT "selection dilution" by non-VBS-like topologies
    lumped into the inclusive sample -- applying this exact cut barely
    changes the trend (see the step-3 Aside), which rules that hypothesis
    out. The actual mechanism: m_jj (tagging-jet-pair mass) is only
    weakly correlated event-by-event with m_WW (the true partonic energy
    scale the operator couples to, see ww_mass()) in this inclusive,
    fixed-order sample -- m_jj tracks Delta eta_jj far more than it
    tracks m_WW. The Aside works this out quantitatively. This selection
    is kept regardless, as the realistic VBS fiducial region: m_WW isn't
    reconstructable in real (neutrino-underconstrained) data, so m_jj
    remains the right observable for an actual analysis even though it's
    a weak per-event tracer of the EFT growth in this sample.

    Any event missing the required objects (< 2 jets or < 2 charged
    leptons) fails, matching the None-propagation convention used by the
    observables above.

    Pass as `selection=` to read_yields()/load_sample() to apply this
    BEFORE the primary observable (dijet_mass by default) is computed and
    binned, so every downstream quantity (bins, theta_pb, ...) describes
    exactly this fiducial region."""
    mjj = dijet_mass(event)
    if mjj is None or mjj <= mjj_min:
        return False
    deta = dijet_deltaeta(event)
    if deta is None or deta <= deltaeta_jj_min:
        return False
    mll = dilepton_mass(event)
    if mll is None or mll <= mll_min:
        return False
    pts = lepton_pts(event)
    if pts is None or pts[0] <= ptl_min or pts[1] <= ptl_min:
        return False
    if missing_et(event) <= met_min:
        return False
    return True


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

    IMPORTANT normalization: event.weights[name] -- the per-event entry in
    the <rwgt> block for benchmark `name`, i.e. what's actually summed
    below -- is generally NOT the same for every event. This sample's own
    param_card.dat generates at cW=cHW=cHWB=cHj3=cHl3=1 rather than at the
    SM point, so even the 'SM' benchmark's weight is an event-by-event
    importance-reweighting correction back to c=0, not a constant
    (verified directly against this exercise's own LHE file: event.weights
    ['SM'] varies by more than two orders of magnitude across the first
    handful of events). The one raw quantity that IS flat across every
    event is XWGTUP itself -- exposed by pylhe as event.eventinfo.weight,
    identical to the <init> block's XSECUP bit-for-bit for every event --
    but that field is the *generation*-point's total cross section and is
    not what's summed here.

    Either way, the standard unbiased Monte Carlo estimator for a
    benchmark point's cross section in a bin is

        sigma_bin(name) = (1/n_total) * sum_{selected events in bin} event.weights[name],

    i.e. divide by the TOTAL number of generated events, not just the ones
    landing in the bin -- this holds regardless of whether weights[name]
    is flat, and is the same convention used in
    LLR_MCinHEP/HEP_differential_xsec.ipynb (`xsec =
    sum(event.eventinfo.weight)/nevents`), generalized from that
    notebook's flat generation weight to the (generally non-flat)
    reweighted benchmarks used here. Skipping the division -- i.e. just
    summing raw weights -- overshoots every yield by a factor of n_total.
    The two accumulators below are therefore divided by n_total (sumw2 by
    n_total**2, since it's a sum of squared per-event weights) right
    before returning. Applying the division here (once, after the
    accumulation loop) rather than per-event inside the loop is
    numerically equivalent (n_total is a fixed constant) and avoids a
    second pass over the file.
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
#    solvable by construction.
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
    points defaults to reweight_points.fit_points() (the 21-point exact
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
    trusting anything built on top of the surface."""
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
    (1 pb = 1000 fb, hence the 1000 factor to go from fb^-1 to pb^-1)."""
    return theta_pb * (lumi_fb * 1000.0)


def asimov_variance(mu0_counts: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Poisson variance of the Asimov dataset (data == c=0 expectation
    exactly, by construction -- see chi2_factory). `floor` keeps
    near-empty high-m_jj tail bins from causing a divide-by-zero."""
    return np.maximum(mu0_counts, floor)


def lnN_vector(mu0_counts: np.ndarray, delta_lnN: float | None) -> np.ndarray | None:
    """Rank-1 vector v = delta_lnN * mu0_counts for a single, fully-
    CORRELATED (across every bin) log-normal-style normalization
    systematic of fractional size delta_lnN (e.g. delta_lnN=0.10 <-> a 10%
    "lnN" uncertainty in the ATLAS/CMS/HistFactory/Combine sense), in the
    usual Gaussian ("asymptotic") approximation theta~N(0,1),
    mu_i(c,theta) ~= mu_i(c) * (1 + delta_lnN * theta).

    v is fixed at the NOMINAL (c=0, SM) Asimov prediction mu0_counts, not
    re-evaluated at whatever c is being tested/fitted -- exactly the same
    "Sigma FIXED" convention already used for sigma2 itself (see
    coverage_scan()'s docstring): the full bin-to-bin covariance this
    systematic induces is Sigma = diag(sigma2) + v v^T, and every function
    below that accepts an `lnn_v` argument profiles it out in closed form
    via weighted_dot()'s Sherman-Morrison update rather than re-fitting a
    nuisance parameter numerically.

    delta_lnN <= 0 (including the default None) returns None -- the "lnN
    off" sentinel every `lnn_v=` argument below understands as "use the
    plain diagonal sigma2, no correlated systematic", so this feature is
    fully opt-in and every existing call site is unaffected until a
    positive delta_lnN is actually passed in.

    THE knob this exercise's step 3 tunes: growing delta_lnN inflates the
    effective uncertainty on the overall normalization, which drowns out
    the (normalization-like, at small c) linear EFT term faster than the
    (shape-changing) quadratic term -- so, at FIXED luminosity, a large
    enough delta_lnN pushes an otherwise linear-dominated operator into
    quadratic-dominated territory. See linear_vs_quadratic_ratio()."""
    if delta_lnN is None or delta_lnN <= 0:
        return None
    return delta_lnN * mu0_counts


# --------------------------------------------------------------------------
# 4. Fisher matrix / PCA / linear-vs-quadratic diagnostics
#
#    Every function below accepts an optional `lnn_v` (see lnN_vector()):
#    lnn_v=None (the default) is the plain diagonal-sigma2 treatment, exactly
#    as before this feature existed; lnn_v=v activates the fully-correlated
#    lnN systematic via weighted_dot()'s Sherman-Morrison profiling.
# --------------------------------------------------------------------------

def weighted_dot(a: np.ndarray, b: np.ndarray, sigma2: np.ndarray,
                  lnn_v: np.ndarray | None = None) -> float:
    """a^T Sigma^-1 b for Sigma = diag(sigma2) (lnn_v=None) or the rank-1-
    correlated Sigma = diag(sigma2) + lnn_v (outer) lnn_v (lnn_v given, see
    lnN_vector()) -- the single building block every Fisher/chi2/cubic-
    solver/coverage_scan() computation below is built from, so that adding
    the optional lnN systematic never means touching more than this one
    function plus the small number of plain `sum(a*b/sigma2)`-style call
    sites that now read `weighted_dot(a, b, sigma2, lnn_v)` instead.

    lnn_v=None reduces EXACTLY to sum(a*b/sigma2) (no Sherman-Morrison term
    computed at all), so every caller is bit-for-bit unchanged when the
    lnN feature is off.

    With lnn_v=v given, uses the Sherman-Morrison-Woodbury rank-1 update

        (D + v v^T)^-1 = D^-1 - (D^-1 v)(D^-1 v)^T / (1 + v^T D^-1 v)

    to profile the shared normalization nuisance out in closed form:

        a^T Sigma^-1 b = sum(a*b/sigma2)
                          - sum(a*Dinv_v) * sum(b*Dinv_v) / denom

    with Dinv_v = v/sigma2 and denom = 1 + sum(v*Dinv_v). Because this is
    still just a fixed symmetric bilinear form (only the diagonal-vs-rank-1
    *shape* of Sigma changed, not the fact that chi2(c) = <r(c),r(c)> for
    r(c) quadratic in c), chi2(c) stays EXACTLY quartic in c with lnN on,
    so _min_chi2_1op()'s closed-form cubic solver needs no structural
    change -- only its 4 coefficients get computed with this weighted_dot()
    instead of a plain diagonal sum."""
    base = np.sum(a * b / sigma2)
    if lnn_v is None:
        return float(base)
    Dinv_v = lnn_v / sigma2
    denom = 1.0 + np.sum(lnn_v * Dinv_v)
    return float(base - np.sum(a * Dinv_v) * np.sum(b * Dinv_v) / denom)


def linear_fisher_matrix(theta_counts: np.ndarray, sigma2: np.ndarray,
                          operators=rwp.OPERATORS,
                          lnn_v: np.ndarray | None = None) -> np.ndarray:
    """F_ij = sum_bins (dN_bin/dc_i)(dN_bin/dc_j) / sigma_bin^2, evaluated
    at c=0 -- the standard linearized EFT Fisher information matrix (the
    Hessian/2 of the Gaussian chi2 at its minimum). Built directly from
    theta_counts' *linear* rows (index 1..n_ops; row 0 is the intercept,
    quadratic/cross rows are not needed for this local, linear-order
    object).

    lnn_v (see lnN_vector()) generalizes this to F_ij = A_i^T Sigma^-1 A_j
    via the same Sherman-Morrison update as weighted_dot(), applied here in
    matrix form (one rank-1 correction shared by every (i,j) pair, rather
    than n_ops^2 separate weighted_dot() calls) for speed."""
    n_ops = len(operators)
    A = theta_counts[1:1 + n_ops, :]     # (n_ops, n_bins) = dN/dc_i per bin
    F = (A / sigma2) @ A.T                # (n_ops, n_ops)
    if lnn_v is None:
        return F
    Dinv_v = lnn_v / sigma2
    denom = 1.0 + np.sum(lnn_v * Dinv_v)
    u = A @ Dinv_v                        # (n_ops,)
    return F - np.outer(u, u) / denom


def quadratic_fisher_diag(theta_counts: np.ndarray, sigma2: np.ndarray,
                           operators=rwp.OPERATORS,
                           lnn_v: np.ndarray | None = None) -> np.ndarray:
    """F^Q_ii = sum_bins (d^2N_bin/dc_i^2)^2 / sigma_bin^2 -- the diagonal
    quadratic-term analogue of linear_fisher_matrix()'s F_ii, built from
    theta_counts' diagonal QUADRATIC rows (index 1+n_ops .. 1+2n_ops-1)
    instead of the linear ones. Only the diagonal is used in this exercise
    -- see linear_vs_quadratic_ratio().

    lnn_v (see lnN_vector()) generalizes each F^Q_ii to Q_i^T Sigma^-1 Q_i,
    same Sherman-Morrison update as linear_fisher_matrix(), diagonal-only."""
    n_ops = len(operators)
    Q = theta_counts[1 + n_ops:1 + 2 * n_ops, :]   # (n_ops, n_bins) = d^2N/dc_i^2 per bin
    base = np.sum(Q ** 2 / sigma2, axis=1)          # (n_ops,)
    if lnn_v is None:
        return base
    Dinv_v = lnn_v / sigma2
    denom = 1.0 + np.sum(lnn_v * Dinv_v)
    u = Q @ Dinv_v                                  # (n_ops,)
    return base - u ** 2 / denom


def linear_vs_quadratic_shape_ratio(theta_pb: np.ndarray,
                                     operators=rwp.OPERATORS) -> np.ndarray:
    """R_i^shape = sqrt(sum_bins Q_i(bin)^2) / sqrt(sum_bins L_i(bin)^2),
    i.e. the (unweighted, L2-norm-over-bins) size of operator i's
    quadratic term relative to its linear term, read directly off
    theta_pb -- the pb-unit response surface, BEFORE scale_theta_to_counts()
    multiplies every term by the same lumi_fb-dependent constant.

    Because that constant cancels exactly in a ratio of two theta_pb norms,
    R_i^shape does not depend on lumi_fb at all: it is a fixed property of
    how operator i deforms the shape of the m_jj spectrum (at the fixed
    reference point c_i=1, everything else 0), not of how much data you
    plan to collect. This is the right tool for step 5's "is this
    operator's EFT response intrinsically linear or quadratic?" question.

    R_i^shape << 1  -> "linear-dominated"
    R_i^shape ~  1  -> "comparable"
    R_i^shape >> 1  -> "quadratic-dominated"

    Contrast with linear_vs_quadratic_ratio() below: that one weighs each
    bin by 1/sigma_bin^2 (Fisher information) and evaluates the ratio at
    each operator's own statistics-dependent 1-sigma point, so it answers
    a genuinely different, luminosity-DEPENDENT question ("does the
    quadratic term matter within the confidence interval this dataset will
    actually deliver?") and can legitimately disagree with R_i^shape for an
    operator whose quadratic term is concentrated in low-statistics bins
    (e.g. a high-m_jj tail) -- see the notebook (step 5) for the worked
    comparison of the two on this exercise's 5 operators."""
    n_ops = len(operators)
    L = theta_pb[1:1 + n_ops, :]                 # (n_ops, n_bins) = dsigma/dc_i, pb
    Q = theta_pb[1 + n_ops:1 + 2 * n_ops, :]      # (n_ops, n_bins) = d^2sigma/dc_i^2, pb
    # A genuinely zero linear term gives R_i^shape = +inf -- semantically
    # correct ("infinitely quadratic-dominated"), so silence the warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.sum(Q ** 2, axis=1)) / np.sqrt(np.sum(L ** 2, axis=1))


def linear_vs_quadratic_ratio(theta_counts: np.ndarray, sigma2: np.ndarray,
                               operators=rwp.OPERATORS,
                               lnn_v: np.ndarray | None = None) -> np.ndarray:
    """R_i = (quadratic term's Fisher-weighted norm) / (linear term's
    Fisher-weighted norm), evaluated AT c_i = sigma_lin,i = 1/sqrt(F_ii)
    (the linear-only 1-sigma point for operator i, everything else at 0):

        R_i = sigma_lin,i * sqrt(F^Q_ii) / sqrt(F_ii)  ==  sqrt(F^Q_ii) / F_ii

    R_i << 1  -> the linear term dominates the response at the scale that
                 actually matters (its own 1-sigma point) -> "linear-dominated"
    R_i ~  1  -> linear and quadratic contribute comparably  -> "comparable"
    R_i >> 1  -> the quadratic term dominates                -> "quadratic-dominated"

    Unlike linear_vs_quadratic_shape_ratio() above, this ratio is built
    from theta_counts = scale_theta_to_counts(theta_pb, lumi_fb) and from
    sigma2 (the Asimov variance at that same lumi_fb), and it is NOT
    luminosity-independent: F_lin and F_quad both scale like lumi_fb, but
    sigma_lin,i = 1/sqrt(F_ii) shrinks like 1/sqrt(lumi_fb), so
    R_i ~ sigma_lin,i * sqrt(F^Q_ii) inherits a net 1/sqrt(lumi_fb)
    scaling -- more data makes EVERY operator look more linear by this
    metric, not because the operator's response changed shape, but because
    the 1-sigma point being probed shrinks. It also inherits an implicit
    per-bin weight ~1/sigma_bin^2 ~ 1/mu0_bin, which privileges the
    highest-statistics (typically lowest-m_jj) bins over low-statistics
    tail bins regardless of lumi_fb. Useful for step 6 ("does the
    quadratic term actually matter at the CI this dataset delivers?"), but
    not the tool to reach for when classifying an operator's intrinsic
    linear/quadratic character -- use linear_vs_quadratic_shape_ratio()
    for that (see the notebook, step 5, for a worked comparison of the
    two on this exercise's 5 operators, and where they agree/disagree and
    why).

    This is a *convention* for ranking/grouping operators, not a uniquely
    "correct" physical threshold -- see the notebook (step 5) for the
    actual numbers and where the lines get drawn for this exercise's 5
    operators.

    lnn_v (see lnN_vector()) generalizes F_lin/F_quad to the lnN-inflated
    covariance via linear_fisher_matrix()/quadratic_fisher_diag()'s own
    Sherman-Morrison update. The Sherman-Morrison correction to F_ii is
    largest for whichever term's bin-by-bin shape is most correlated (in
    the 1/sigma2-weighted sense) with the nominal SM shape mu0 -- i.e. for
    whichever term looks most like a pure normalization shift, exactly
    what a correlated lnN systematic marginalizes over. For this
    exercise's operators the linear term is closer to a normalization
    shift and the quadratic term a genuine shape distortion concentrated
    in the high-m_T^WW tail (see step 2), so growing delta_lnN at fixed
    lumi_fb erodes F_lin far more than F_quad and pushes R_i up -- the
    mechanism step 3 uses to move an operator from linear- to quadratic-
    dominated without touching the luminosity at all."""
    F_lin = np.diag(linear_fisher_matrix(theta_counts, sigma2, operators, lnn_v))
    F_quad = quadratic_fisher_diag(theta_counts, sigma2, operators, lnn_v)
    # An operator with a genuinely (numerically) zero linear term gives
    # F_lin_ii = 0 -- R_i = +inf is the semantically correct answer
    # ("infinitely quadratic-dominated"), not an error, so silence the
    # resulting divide-by-zero warning rather than let it alarm students.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(F_quad) / F_lin


def eigh_sorted(F: np.ndarray):
    """Eigen-decomposition of a symmetric matrix, sorted by DESCENDING
    eigenvalue. Returns (eigvals[n], eigvecs[n,n]) with eigvecs[:, k] the
    k-th eigenvector (unit-normalized, numpy convention). Not central to
    this exercise (no PCA/flat-direction step here), kept for convenience/
    consistency with ../flat_directions/fitlib.py."""
    w, v = np.linalg.eigh(F)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def correlation_matrix(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def chi2_curve_1d(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                   op: str, grid: np.ndarray, fixed: dict | None = None,
                   operators=rwp.OPERATORS,
                   lnn_v: np.ndarray | None = None) -> np.ndarray:
    """Scan a single operator over `grid`, holding every other operator at
    exactly 0 except whatever is explicitly named in `fixed`. This is the
    "individually" fit this exercise's step 4 asks for -- call it once with
    the full theta_counts and once with theta_linear_only(theta_counts) to
    get the "including" / "excluding the quadratic EFT component" curves.

    lnn_v (see lnN_vector()) switches each point's chi2 evaluation from the
    plain diagonal sum(r^2/sigma2) to weighted_dot()'s Sherman-Morrison-
    profiled version -- the curve stays a well-defined chi2(c) either way,
    just against a wider (correlated-systematic-inflated) effective
    covariance."""
    fixed = {} if fixed is None else fixed
    chi2 = np.empty_like(grid, dtype=float)
    for k, val in enumerate(grid):
        cvals = dict(fixed)
        cvals[op] = val
        mu = predict(theta_counts, cvals, operators)
        r = mu - mu0
        chi2[k] = weighted_dot(r, r, sigma2, lnn_v)
    return chi2


def sigma_from_curve(grid: np.ndarray, chi2vals: np.ndarray, delta: float = 1.0) -> float:
    """Read a 1-sigma uncertainty directly off a 1D chi2(c) curve: walk
    outward from the curve's own minimum in both directions until it
    crosses chi2_min + delta (delta=1 <-> the standard 68% CL interval for
    one parameter), linearly interpolating between grid points, and return
    half the total width."""
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
    isn't available."""
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
                  operators=rwp.OPERATORS, lnn_v: np.ndarray | None = None):
    """Build a vector-call chi2(c) for iminuit. mu0 IS the c=0 (SM)
    prediction exactly (Asimov technique: "data" has no statistical
    fluctuation), so c=0 is guaranteed to be a global minimum with chi2=0.

    lnn_v (see lnN_vector()), if given, routes chi2 through weighted_dot()
    instead of a plain diagonal sum, so a Minuit cross-check (fit_full())
    sees exactly the same Sherman-Morrison-profiled lnN systematic as the
    closed-form solver in section 6."""
    def chi2(c):
        cvals = dict(zip(operators, c))
        mu = predict(theta_counts, cvals, operators)
        r = mu - mu0
        return weighted_dot(r, r, sigma2, lnn_v)
    return chi2


def _run_migrad_hesse_minos(m: Minuit) -> Minuit:
    m.migrad()
    m.hesse()
    try:
        m.minos()
    except Exception as exc:
        print(f"    [minos] raised {type(exc).__name__}: {exc}")
    return m


def fit_full(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
             operators=rwp.OPERATORS, start=None, limit: float | None = 5.0,
             fixed: list[str] | None = None,
             lnn_v: np.ndarray | None = None) -> Minuit:
    """Fit all len(operators) Wilson coefficients simultaneously, MIGRAD-
    initialized at k=0 (SM). `fixed`, if given, pins the named operators to
    0 instead of leaving them free -- e.g. fixed=[o for o in operators if
    o != "cW"] gives exactly an "individual" 1-operator fit, usable as a
    Minuit cross-check of chi2_curve_1d()+sigma_from_curve(). lnn_v (see
    lnN_vector()), if given, is forwarded to chi2_factory()."""
    chi2 = chi2_factory(theta_counts, mu0, sigma2, operators, lnn_v)
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


# --------------------------------------------------------------------------
# 6. Toy-MC Delta-chi^2=1 coverage vs. Wilks' theorem (step 6)
#
#    Single-operator study throughout (every other operator fixed at 0,
#    matching chi2_curve_1d()'s "individual" fits above): for operator op
#    with linear row L=theta[1+i,:] and diagonal-quadratic row
#    Q=theta[1+n_ops+i,:], the model is exactly quadratic in the one free
#    coefficient c:
#
#        mu(c) = base + c*L + c^2*Q          (base = theta[0,:], the SM row)
#
#    A single Gaussian pseudo-experiment ("toy") at assumed-true c_true is
#    data = mu(c_true) + Normal(0, sigma2), using the FIXED (c-independent)
#    Asimov sigma2 -- exactly arXiv:2207.01350's own toy recipe ("samples
#    from a Gaussian with mean mu(c_true) and fixed covariance Sigma"),
#    deliberately isolating the effect the paper studies (Wilks' theorem
#    breaking from the model being nonlinear in c) from any separate
#    Poisson-small-count effect.
#
#    chi2(c; data) = sum_bins (base + c*L + c^2*Q - data)^2 / sigma2 is
#    QUARTIC in c (mu is quadratic, chi2 squares it), so its minimum is
#    found by solving dchi2/dc=0, a CUBIC in c, in closed form (see
#    _min_chi2_1op) -- exact, and far faster than an iterative Minuit call
#    per toy. The notebook cross-checks this against Minuit on a handful of
#    toys before trusting it for the full scan.
#
#    Every function below also accepts an optional `lnn_v` (see
#    lnN_vector()): each plain diagonal sum(x*y/sigma2) becomes
#    weighted_dot(x, y, sigma2, lnn_v) instead. Because weighted_dot() is
#    still just a FIXED symmetric bilinear form (Sherman-Morrison changes
#    which form, not the fact that it's bilinear), chi2(c) stays exactly
#    quartic in c with lnN on, so the cubic-root solver's structure is
#    untouched -- only its 4 coefficients change. Toys then need a
#    two-part noise model to match: a shared per-toy pull along lnn_v (the
#    correlated systematic) plus the existing independent per-bin
#    statistical noise -- see coverage_scan().
# --------------------------------------------------------------------------

def _chi2_1op(d: np.ndarray, L: np.ndarray, Q: np.ndarray, sigma2: np.ndarray,
              c: float, lnn_v: np.ndarray | None = None) -> float:
    """chi2(c) for a single free operator, given d = base - data (per bin)."""
    r = d + c * L + c ** 2 * Q
    return weighted_dot(r, r, sigma2, lnn_v)


def _min_chi2_1op(d: np.ndarray, L: np.ndarray, Q: np.ndarray, sigma2: np.ndarray,
                   lnn_v: np.ndarray | None = None):
    """Global minimum of _chi2_1op(d, L, Q, sigma2, c, lnn_v) over c, found
    exactly via calculus (no iterative fit needed) -- by the product rule,
    the same way you'd do it by hand.

    Derivation
    ----------
    Per bin the model is mu(c) = base + c*L + c^2*Q, so the residual
    r(c) = mu(c) - data = d + c*L + c^2*Q            (d := base - data,
                                                        c-INdependent: this
                                                        is where "data and
                                                        the SM don't depend
                                                        on c" enters)
    is itself a quadratic in c, with derivative
        dr/dc = L + 2*c*Q .
    chi2(c) = <r(c), r(c)>_W, where <a, b>_W = weighted_dot(a, b, sigma2,
    lnn_v) is the plain diagonal sum(a*b/sigma2) when lnn_v=None, or its
    Sherman-Morrison-profiled generalization when lnn_v is given -- either
    way a FIXED bilinear form. The product rule then gives

        dchi2/dc = 2 <r(c), dr/dc>_W = 2 <d + c*L + c^2*Q,  L + 2*c*Q>_W .

    (d's own derivative is what vanishes, not d itself -- d survives
    inside r(c) and still multiplies the L and 2cQ pieces of dr/dc below.)
    Expanding this bilinear pairing and collecting powers of c gives
    exactly the CUBIC dchi2/dc=0 solved below:

        dchi2/dc = 2*(A0 + A1 c + A2 c^2 + A3 c^3) = 0   where
            A0 = <d, L>_W
            A1 = 2 <d, Q>_W + <L, L>_W
            A2 = 3 <L, Q>_W
            A3 = 2 <Q, Q>_W

    equivalently: chi2(c) is the quartic sum_{a,b in {0,1,2}} M[a,b] c^(a+b)
    built from the symmetric "Gram matrix" M[a,b] = <r_a, r_b>_W of r(c)'s
    own coefficients r0=d, r1=L, r2=Q (bilinearity of <,>_W is why chi2(c)
    is exactly a polynomial in c at all); A0=M[0,1], A1=2M[0,2]+M[1,1],
    A2=3M[1,2], A3=2M[2,2] -- the same 5 dot products, named to match the
    code below one-for-one. This cubic has exactly the same shape whether
    <,>_W is the plain diagonal form or the lnN-profiled one, because
    bilinearity is all the derivation above uses -- only the coefficients'
    VALUES change.

    Special case worth knowing (does not change the code path below, just
    explains a root you may notice): if d=0 EXACTLY -- true only when
    data==base itself, e.g. evaluating the Asimov curve at its own true
    point -- then A0=<0,L>_W=0 and c=0 is an exact analytic root (r(0)=d=0
    already gives chi2(0)=0, the global minimum, trivially). For a SINGLE
    bin the cubic even factors visibly in that case:
        dchi2/dc = 2c (cQ+L)(2cQ+L),   roots c=0, c=-L/Q, c=-L/(2Q).
    But that per-bin factoring does not survive summing over >1 bin: c=0
    remains an exact root (every term in the sum still carries an explicit
    overall factor of c), while the other two roots come from the
    QUADRATIC FORMULA applied to the already-bin-SUMMED A1, A2, A3 above --
    not from any single bin's L/Q, since different bins generically have
    different L_i/Q_i (that bin-to-bin difference in shape is exactly the
    extra information a shape fit has over a rate-only one). So there's no
    shortcut around solving the summed cubic in general -- and in this
    codebase d is essentially never exactly 0 anyway, since
    _min_chi2_1op() is always called against real/toy data, never against
    data==base (see min_chi2_1op(), coverage_scan(), toy_q_distribution()).
    A0..A3 below are therefore kept fully general rather than special-cased
    for d=0.

    Evaluate chi2 at every real root of that cubic and keep the smallest --
    correct regardless of whether the quartic has 1 or 2 local minima
    (its leading coefficient <Q,Q>_W >= 0 always, so chi2 -> +inf
    as c -> +-inf, and a real cubic always has >= 1 real root, so this
    never comes up empty).

    Returns (c_hat, chi2_min).
    """
    # Five weighted dot products, named by (a, b) power-of-c index into the
    # r(c)=d+cL+c^2Q "Gram matrix" M[a,b]=<r_a,r_b>_W above (r0=d,r1=L,r2=Q).
    M01 = weighted_dot(d, L, sigma2, lnn_v)    # <d, L>_W
    M02 = weighted_dot(d, Q, sigma2, lnn_v)    # <d, Q>_W
    M11 = weighted_dot(L, L, sigma2, lnn_v)    # <L, L>_W
    M12 = weighted_dot(L, Q, sigma2, lnn_v)    # <L, Q>_W
    M22 = weighted_dot(Q, Q, sigma2, lnn_v)    # <Q, Q>_W

    # dchi2/dc = 2*(A0 + A1 c + A2 c^2 + A3 c^3); np.roots wants the highest
    # power first, i.e. [A3, A2, A1, A0].
    A0 = M01
    A1 = 2.0 * M02 + M11
    A2 = 3.0 * M12
    A3 = 2.0 * M22

    roots = np.roots([A3, A2, A1, A0])
    tol = 1e-6 * np.maximum(np.abs(roots.real), 1.0)
    real_roots = roots.real[np.abs(roots.imag) < tol]
    if real_roots.size == 0:                      # pragma: no cover -- degenerate fallback
        real_roots = roots.real

    chi2_vals = np.array([_chi2_1op(d, L, Q, sigma2, c, lnn_v) for c in real_roots])
    kbest = int(np.argmin(chi2_vals))
    return float(real_roots[kbest]), float(chi2_vals[kbest])


def min_chi2_1op(theta_counts: np.ndarray, mu0: np.ndarray, sigma2: np.ndarray,
                  op: str, data: np.ndarray, operators=rwp.OPERATORS,
                  lnn_v: np.ndarray | None = None):
    """Public wrapper around _min_chi2_1op: minimize chi2(c; data) over the
    single free operator `op` (all others fixed at 0), for an arbitrary
    per-bin `data` array (mu0 is unused directly -- kept in the signature
    for symmetry with chi2_factory()/fit_full(); base = theta_counts[0,:]
    is what "d = base - data" actually needs). Returns (c_hat, chi2_min)."""
    n_ops = len(operators)
    i = operators.index(op)
    base = theta_counts[0, :]
    L = theta_counts[1 + i, :]
    Q = theta_counts[1 + n_ops + i, :]
    d = base - data
    return _min_chi2_1op(d, L, Q, sigma2, lnn_v)


def coverage_scan(theta_counts: np.ndarray, sigma2: np.ndarray, op: str,
                   c_true_grid: np.ndarray, n_toys: int = 2000, delta: float = 1.0,
                   operators=rwp.OPERATORS, seed: int = 0,
                   lnn_v: np.ndarray | None = None) -> np.ndarray:
    """Toy-MC coverage test for Wilks' theorem (arXiv:2207.01350 methodology
    -- see module docstring above for the full recipe).

    For each c_true in c_true_grid: throw n_toys Gaussian pseudo-datasets
    around mu(c_true), then for each toy compute the profile-likelihood-
    ratio test statistic q(c_true) = chi2(c_true) - chi2(c_hat) (c_hat =
    that toy's own global best fit). "Covered" means q(c_true) <= delta
    (delta=1 <-> the Wilks' Delta-chi^2=1, 68.27%-CL threshold for 1 dof;
    delta=3.84 <-> the 95% threshold, chi2.ppf(0.95, 1) -- any delta works,
    this function just counts how often the true value falls inside the
    corresponding Delta-chi^2 contour).

    lnn_v (see lnN_vector()), if given, switches BOTH halves of the toy
    machinery from the plain-diagonal treatment to the fully-correlated
    lnN one:
      - each toy is drawn from the TWO-part noise model
        data_i = mu_true,i + v_i*theta + sqrt(sigma2_i)*eps_i, theta~N(0,1)
        drawn ONCE per toy (the shared, fully-correlated normalization
        pull) and eps_i drawn independently per bin (the usual statistical
        noise) -- equivalent to drawing from N(mu_true, diag(sigma2) + v v^T)
        without ever building or Cholesky-decomposing the dense n_bins x
        n_bins covariance matrix;
      - each toy's chi2(c_true) and chi2(c_hat) are evaluated with the same
        lnn_v via _chi2_1op()/_min_chi2_1op(), i.e. with theta profiled out
        in closed form at both c_true and c_hat, not held fixed at its
        true-toy value -- exactly the "asymptotic profiling" a real
        lnN-nuisance fit does.

    Returns coverage: array[len(c_true_grid)] = fraction of toys with
    q(c_true) <= delta at each true value. Compare directly to the 68.27%
    Wilks target -- if it isn't flat at 68.27%, Wilks' theorem is violated
    at that c_true.
    """
    rng = np.random.default_rng(seed)
    n_ops = len(operators)
    i = operators.index(op)
    base = theta_counts[0, :]
    L = theta_counts[1 + i, :]
    Q = theta_counts[1 + n_ops + i, :]
    sqrt_sigma2 = np.sqrt(sigma2)
    n_bins = len(base)

    coverage = np.empty(len(c_true_grid), dtype=float)
    for k, c_true in enumerate(c_true_grid):
        mu_true = base + c_true * L + c_true ** 2 * Q
        if lnn_v is None:
            toys = mu_true[None, :] + rng.normal(size=(n_toys, n_bins)) * sqrt_sigma2[None, :]
        else:
            theta_pull = rng.normal(size=(n_toys, 1))               # shared per-toy lnN pull
            eps = rng.normal(size=(n_toys, n_bins)) * sqrt_sigma2[None, :]
            toys = mu_true[None, :] + theta_pull * lnn_v[None, :] + eps
        n_covered = 0
        for row in toys:
            d = base - row
            c_hat, chi2_min = _min_chi2_1op(d, L, Q, sigma2, lnn_v)
            chi2_true = _chi2_1op(d, L, Q, sigma2, c_true, lnn_v)
            if (chi2_true - chi2_min) <= delta:
                n_covered += 1
        coverage[k] = n_covered / n_toys
    return coverage


def toy_q_distribution(theta_counts: np.ndarray, sigma2: np.ndarray, op: str,
                        c_true: float = 0.0, n_toys: int = 20000,
                        operators=rwp.OPERATORS, seed: int = 0,
                        lnn_v: np.ndarray | None = None) -> np.ndarray:
    """The single-`c_true` building block behind coverage_scan(): draw
    `n_toys` Gaussian pseudo-datasets around mu(c_true) (same recipe, same
    FIXED Asimov sigma2 -- see coverage_scan()'s docstring) and return the
    raw array of `n_toys` profile-likelihood-ratio values
    q_k = chi2(c_true; toy_k) - chi2(c_hat_k; toy_k), instead of collapsing
    them straight to a single coverage fraction the way coverage_scan()
    does for a whole c_true_grid at one fixed delta.

    Exposing the raw distribution (rather than just coverage(delta)) is
    what the notebook's toy-CALIBRATED confidence-band step needs:
    coverage_scan() answers "what fraction of toys satisfy q<=delta, for a
    delta chosen ahead of time (Wilks' asymptotic delta=1 or delta=3.84)";
    this function answers the complementary question "what delta would I
    need, empirically, for exactly 68.3%/95% of toys to satisfy q<=delta" --
    i.e. it lets you read the toy-calibrated delta straight off
    np.quantile(q, target_CL) instead of assuming Wilks' asymptotic chi2_1
    value is the right threshold. See toy_calibrated_delta() for that last
    step, and coverage_scan()'s docstring for the two-part lnN toy noise
    model (`lnn_v`) shared by both functions."""
    rng = np.random.default_rng(seed)
    n_ops = len(operators)
    i = operators.index(op)
    base = theta_counts[0, :]
    L = theta_counts[1 + i, :]
    Q = theta_counts[1 + n_ops + i, :]
    sqrt_sigma2 = np.sqrt(sigma2)
    n_bins = len(base)

    mu_true = base + c_true * L + c_true ** 2 * Q
    if lnn_v is None:
        toys = mu_true[None, :] + rng.normal(size=(n_toys, n_bins)) * sqrt_sigma2[None, :]
    else:
        theta_pull = rng.normal(size=(n_toys, 1))               # shared per-toy lnN pull
        eps = rng.normal(size=(n_toys, n_bins)) * sqrt_sigma2[None, :]
        toys = mu_true[None, :] + theta_pull * lnn_v[None, :] + eps

    q = np.empty(n_toys, dtype=float)
    for k, row in enumerate(toys):
        d = base - row
        c_hat, chi2_min = _min_chi2_1op(d, L, Q, sigma2, lnn_v)
        chi2_true = _chi2_1op(d, L, Q, sigma2, c_true, lnn_v)
        q[k] = chi2_true - chi2_min
    return q


def toy_calibrated_delta(q: np.ndarray, cl: float = 0.6826894921370859) -> float:
    """The empirical Delta-chi^2 threshold that gives exactly `cl` toy
    coverage, read directly off the q-distribution from
    toy_q_distribution() as its `cl` quantile (q<=this value for a `cl`
    fraction of the toys, by construction/definition of the quantile) --
    the toy-calibrated counterpart to the FIXED Wilks asymptotic delta (1.0
    for 68.27%, chi2.ppf(0.95,1)=3.841... for 95%). Comparing the two
    directly -- same c_true, same operator, same dataset -- is exactly "is
    Wilks' theorem's assumed Delta-chi^2 threshold the right one to use
    here, or does this operator's own nonlinearity need a different one":
    toy_calibrated_delta() > the Wilks value means the naive Delta-chi^2
    interval at that CL is too NARROW (under-covers) and should be widened;
    toy_calibrated_delta() < the Wilks value means it's too WIDE
    (over-covers)."""
    return float(np.quantile(q, cl))


# --------------------------------------------------------------------------
# 7. One-call pipeline
# --------------------------------------------------------------------------

def load_sample(lhe_path, bins: np.ndarray, lumi_fb: float = 300.0,
                 observable=dijet_mass, selection=None,
                 operators=rwp.OPERATORS) -> dict:
    """Read one LHE(.gz) sample and run steps 1-4 above in one call. Returns
    a dict with everything steps 5+ (Fisher/classification/toys, done
    separately so you can inspect/plot intermediate results) need:

        bins, sumw, sumw2, n_total          -- raw histograms (pb)
        theta_pb                            -- quadratic surface, pb
        validation                          -- validate_surface() rows
        theta                               -- quadratic surface, expected counts @ lumi_fb
        mu0, sigma2                         -- Asimov data & variance (counts)

    `selection`, if given (e.g. vbs_selection), is applied per-event
    before `observable` -- see read_yields()."""
    fit_pts = rwp.fit_points()
    val_pts = rwp.validation_points()
    all_names = [n for n, _ in fit_pts] + [n for n, _ in val_pts]

    sumw, sumw2, n_total = read_yields(lhe_path, bins, all_names,
                                        observable=observable, selection=selection)
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
