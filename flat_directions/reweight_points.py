#!/usr/bin/env python3
"""
reweight_points.py -- single source of truth for the 6-operator Wilson-
coefficient benchmark grid used throughout the "flat directions" lecture
(WW -> e mu + MET, and its closing process ZZ -> e+ e- mu+ mu-; see
create_ww.sh / create_zz.sh). CC Drell-Yan (create_ccdy.sh) also reuses this
same grid -- it makes a brief cameo in the notebook as a "first guess" for a
closing process that turns out NOT to work (see create_zz.sh's header for the
full reasoning for why ZZ is needed instead).

Both the MadGraph generation scripts (create_ww.sh, create_zz.sh, create_ccdy.sh
-- via `python3 reweight_points.py` printing a ready-to-use reweight_card.dat)
and the analysis notebook (via `import reweight_points` to build the fit
design matrix) read the point list from here, so generation and analysis can
never drift out of sync.

Operator set (SMEFTsim_topU3l_MwScheme_UFO, Block SMEFT indices from
restrict_SMlimit_massless.dat):

    cW    (2)   purely-bosonic triple/quartic gauge operator
    cHWB  (9)   Higgs-W-B mixing -> WWZ/WWgamma (neutral TGC-like)
    cHW   (7)   Higgs-W current  -> WWZ/WWgamma (neutral TGC-like)
    cHj3  (28)  light-quark doublet current     -> qq'W / qqZ production vertex
    cHl3  (104) lepton doublet current          -> W-lepton-neutrino & Z-lepton
                                                    decay vertex, AND (via dGf)
                                                    the universal EW-input shift
    cll1  (107) 4-lepton contact (mu-decay flavour structure) -> 4-lepton
                                                    contact vertices, the dGf
                                                    shift below, AND (see next
                                                    paragraph) a direct term in
                                                    the bare W-lepton-neutrino
                                                    vertex too

Verified directly against parameters.py/couplings.py/vertices.py in this UFO
(not from a paper -- see notebook Section 0 for how to reproduce this check
yourself):
    dGf = (2*cHl3 - cll1) * vevhat**2 / LambdaSMEFT**2          [parameters.py:2185]
    cW, cHW, cHB, cHWB couple ONLY to purely-bosonic vertices (WWZ, WWgamma,
    WWZZ, WWWW, ...) -- zero fermion-line vertices among the 61 coupling
    constants that depend on them (grep-verified against vertices.py).
    cHl3 and cll1 do NOT enter charged- and neutral-current fermion vertices
    the same way, which is exactly why a second, neutral-current process is
    needed to break their degeneracy (see create_zz.sh's header for the full
    argument this feeds into):
      - the bare W-lepton-neutrino vertex carries a cll1 term (GC_507) but NO
        cHl3 term at all;
      - the bare Z-e+e- vertex carries BOTH a cHl3 term (GC_545) and a cll1
        term (GC_548), on the same Lorentz structure -- combining coherently
        as (2*cHl3 - cll1), the same direction as dGf above.
    So in WW (charged current), the only cHl3 dependence at all comes through
    the universal dGf piece, while cll1 gets both the dGf piece AND the direct
    GC_507 piece -- yet empirically (see notebook Step 6) these still combine
    into a near-perfectly degenerate direction for WW. In ZZ (neutral
    current), both operators get the dGf piece AND a direct piece on the same
    vertex, along the same (2,-1) ratio as dGf itself -- and yet ZZ's fitted
    degeneracy direction is measurably *less* degenerate than WW's, not more.
    This residual is a genuine, only partially-understood detail of the full
    coupling structure (mixed photon/Z diagrams, kinematic reweighting of the
    binned Fisher matrix, etc.) that goes beyond a 4h lecture to fully chase
    down; what matters pedagogically -- and what the notebook verifies
    directly from the generated samples, not just asserts -- is that WW and
    ZZ's fitted (cHl3, cll1) directions are different enough to combine into a
    well-constrained fit.

cHB and cHDD are deliberately left out of this headline 6-operator set (kept
to a size that is tractable for a 4h session); they're natural "bonus round"
additions -- see the notebook's closing remarks.
"""
from __future__ import annotations

import random

OPERATORS = ["cW", "cHWB", "cHW", "cHj3", "cHl3", "cll1"]

SMEFT_INDEX = {
    "cW": 2,
    "cHWB": 9,
    "cHW": 7,
    "cHj3": 28,
    "cHl3": 104,
    "cll1": 107,
}

# Operators with literally zero tree-level dimension-6 sensitivity in CC
# Drell-Yan (pp -> l nu): no WWZ/WWgamma vertex exists in single-W production,
# so cW/cHWB/cHW carry no fermion-line coupling that this process could ever
# see (see the module docstring). Used by the notebook to sanity-check the
# CC-DY fit (their fitted rows/columns should come out ~0, not asserted).
CCDY_BLIND_OPERATORS = ["cW", "cHWB", "cHW"]


def fit_points() -> list[tuple[str, dict]]:
    """Benchmark points spanning the exact quadratic response surface

        sigma(c) = sigma_SM + sum_i a_i c_i + sum_{i<=j} b_ij c_i c_j

    in 6 dimensions: 1 (SM) + 6 (linear) + 6 (quadratic diagonal) + 15
    (quadratic cross terms) = 28 unknowns. The point list below supplies
    exactly 28 independent points:
        - SM                                   (1 point,  c = 0)
        - c_i = +1 and c_i = -1, one op at a time   (12 points; +-1 on the
          same op separates the linear term (odd in c_i) from the diagonal
          quadratic term (even in c_i) by finite differences, exactly as
          create.sh's cHWB_0/p1/m1 trick does for the single-operator case)
        - c_i = c_j = +1 for every unique pair i<j   (15 points; needed for
          the cross terms b_ij -- without these, PCA could never see any
          *correlation* between operators, which is precisely where flat
          directions like cHl3-cll1 live)
    """
    pts: list[tuple[str, dict]] = [("SM", {})]
    for op in OPERATORS:
        pts.append((f"{op}_p1", {op: 1.0}))
        pts.append((f"{op}_m1", {op: -1.0}))
    for i, oi in enumerate(OPERATORS):
        for oj in OPERATORS[i + 1:]:
            pts.append((f"{oi}_{oj}_pp", {oi: 1.0, oj: 1.0}))
    return pts


def validation_points(seed: int = 0, n: int = 6, scale: float = 1.5) -> list[tuple[str, dict]]:
    """Extra points NOT used in the quadratic-surface fit -- held out purely
    to check the fitted surface's predictive power at random points in the
    6D space (predicted vs. MG5's own reweighted weight at that point)."""
    rng = random.Random(seed)
    pts = []
    for k in range(n):
        c = {op: round(rng.uniform(-scale, scale), 3) for op in OPERATORS}
        pts.append((f"val{k}", c))
    return pts


def all_points() -> list[tuple[str, dict]]:
    return fit_points() + validation_points()


def reweight_card_text() -> str:
    lines = []
    for name, cvals in all_points():
        lines.append(f"launch --rwgt_name={name}")
        for op in OPERATORS:
            lines.append(f"  set SMEFT {SMEFT_INDEX[op]} {cvals.get(op, 0.0)}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(reweight_card_text())
