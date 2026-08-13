#!/usr/bin/env python3
"""
reweight_points.py -- single source of truth for the 5-operator Wilson-
coefficient benchmark grid used by the "Wilks' theorem" hands-on exercise
(same-sign WW VBS: p p > e+ mu+ ve vm j j QCD=0 SMHLOOP=0 NP=1; see
create_vbs.sh).

This is a direct sibling of ../flat_directions/reweight_points.py, reusing
the exact same SMEFTsim_topU3l_MwScheme_UFO model and the exact same
restrict_ww6op_massless.dat restriction card (6 operators kept external,
everything else pruned to 0) -- we just don't populate one of the six
(cll1) in any reweight point here, so it's never non-zero in this sample.
Reusing the restriction card as-is (rather than cutting a new one) costs
nothing: MadGraph reweighting only ever sets the operators a given
`launch` block explicitly names, so an unused-but-external cll1 is inert.

Operator set (5 of the 6 from the flat-directions lecture; cll1 dropped --
see the top-level planning discussion: keeps one quark-current operator
(cHj3, directly relevant to the VBS-tagging jets) and one lepton-current
operator (cHl3), plus the three purely-bosonic TGC/QGC-type operators
(cW, cHWB, cHW) that also control the quartic vertices probed by VBS. Also
deliberately avoids reintroducing the near-perfectly-degenerate cHl3-cll1
flat direction from the other notebook as a confound in *this* single-
operator-at-a-time exercise):

    cW    (2)   purely-bosonic triple/quartic gauge operator
    cHWB  (9)   Higgs-W-B mixing -> WWZ/WWgamma (neutral TGC-like)
    cHW   (7)   Higgs-W current  -> WWZ/WWgamma (neutral TGC-like)
    cHj3  (28)  light-quark doublet current     -> qq'W / qqZ production vertex
                                                    (couples directly to the
                                                    VBS-tagging jets' quark lines)
    cHl3  (104) lepton doublet current          -> W-lepton-neutrino vertex
                                                    (both W decays here)
"""
from __future__ import annotations

import random

OPERATORS = ["cW", "cHWB", "cHW", "cHj3", "cHl3"]

SMEFT_INDEX = {
    "cW": 2,
    "cHWB": 9,
    "cHW": 7,
    "cHj3": 28,
    "cHl3": 104,
}


def fit_points() -> list[tuple[str, dict]]:
    """Benchmark points spanning the exact quadratic response surface

        sigma(c) = sigma_SM + sum_i a_i c_i + sum_{i<=j} b_ij c_i c_j

    in 5 dimensions: 1 (SM) + 5 (linear) + 5 (quadratic diagonal) + 10
    (quadratic cross terms) = 21 unknowns. The point list below supplies
    exactly 21 independent points -- same finite-difference construction
    as flat_directions/reweight_points.py:
        - SM                                   (1 point,  c = 0)
        - c_i = +1 and c_i = -1, one op at a time   (10 points)
        - c_i = c_j = +1 for every unique pair i<j   (10 points)
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
    5D space (predicted vs. MG5's own reweighted weight at that point)."""
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
