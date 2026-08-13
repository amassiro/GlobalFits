#!/usr/bin/env bash
set -euo pipefail

# create_zz.sh -- generate the multiboson "closing process" for step 7-9 of
# the flat-directions notebook: neutral-current diboson production,
# p p > e+ e- mu+ mu- NP=1, same 6-operator restriction/benchmark grid as
# create_ww.sh (see that script + reweight_points.py for the full
# rationale).
#
# Why ZZ/Z-gamma*->4l (not e.g. Drell-Yan) closes the cHl3-cll1 direction,
# and why it has to be a genuine SECOND multiboson process rather than any
# other process:
#
#   WW->e mu + MET is a purely CHARGED-CURRENT process: both leptonic legs
#   are W-lepton-neutrino vertices. This new process, p p > e+ e- mu+ mu-,
#   is purely NEUTRAL-CURRENT: both leptonic legs are Z/gamma*-lepton-lepton
#   vertices. cHl3 (Higgs-lepton-doublet current) and cll1 (4-lepton
#   contact) both feed into the universal G_F/vevhat shift
#       dGf = (2*cHl3 - cll1)*vevhat**2/LambdaSMEFT**2      [parameters.py:2185]
#   which affects EVERY EW process the same way -- this is exactly what
#   makes cHl3 vs. cll1 individually invisible to WW alone (see the
#   notebook's step 6: WW's sensitivity to {cHl3, cll1} is, to very good
#   approximation, a single line in the (cHl3, cll1) plane along that 2:-1
#   combination -- corr(cHl3, cll1) = 0.998 in the WW-only fit).
#
#   But cHl3 and cll1 ALSO have genuine additional effects on the fermion-
#   boson vertices *beyond* the universal dGf piece, and those extra pieces
#   do NOT carry the same relative cHl3:cll1 weight in a charged-current
#   vertex as in a neutral-current one -- W and Z couple to different
#   components of the lepton-doublet SU(2) current (raising/lowering
#   tau^+/tau^- for W, diagonal tau^3 mixed with hypercharge for Z), so
#   there's no reason for the extra piece to line up the same way. Verified
#   directly against couplings.py this session (grep-reproducible):
#     - the bare (no extra external Higgs leg) W-lepton-neutrino vertex
#       carries a cll1 term (GC_507) but NO cHl3 term at all;
#     - the Z-e+e- vertex carries BOTH a cHl3 term (GC_545) and a cll1 term
#       (GC_548), on the very same Lorentz structure (FFV1) -- i.e. these
#       genuinely add coherently, in a 2*cHl3 : -1*cll1 ratio (same ratio as
#       dGf, incidentally -- the Z vertex's *direct* piece and the
#       *universal* dGf piece reinforce each other here, while for W they
#       partially don't, since W's direct piece is cll1-only).
#   The exact per-vertex coefficients trace back to SMEFTsim's Warsaw-basis
#   field-redefinition/equation-of-motion bookkeeping for redundant
#   operators -- interesting, but a rabbit hole beyond a 4h lecture, and NOT
#   needed to make the point: what matters pedagogically, and what the
#   notebook actually verifies (not just asserts), is that the near-flat
#   {cHl3,cll1} direction fitted from the REAL generated ZZ sample has a
#   different slope than WW's, so combining the two Fisher matrices turns
#   two degenerate lines into a well-constrained crossing point.
#
#   This also had to be a genuine second MULTIBOSON process (fits the
#   school's theme) rather than e.g. single-W Drell-Yan -- and it costs
#   nothing on the cW/cHW/cHWB side: there is no s-channel neutral-TGC
#   (ZZZ/ZZgamma/Zgammagamma) vertex in SMEFT at dimension six (confirmed:
#   0 of cW's 15 and cHW's 20 dependent coupling constants touch any
#   fermion line at all -- purely bosonic operators, exactly as documented
#   in reweight_points.py), so this sample brings no new handle on those
#   three -- but it never needed to: cW/cHW/cHWB were never part of the
#   degenerate direction in the first place, only cHl3/cll1 were.
#
# Final state: p p > e+ e- mu+ mu-, opposite-flavour pairs (e+e-, mu+mu-),
# same reasoning as create_ww.sh's e/mu choice -- avoids identical-particle
# interference/combinatorics that a same-flavour 4e or 4mu final state would
# add (extra diagrams for the second same-flavour pairing), while adding
# nothing physics-wise since cHl3/cll1/cHj3 are lepton-flavour-universal in
# this UFO (see create_ccdy.sh's comment on the same point). Unlike
# create_ww.sh, this final state is already its own charge conjugate, so no
# separate "add process" line is needed.
#
# Reuses the identical 6D basis/restriction card as create_ww.sh and
# create_ccdy.sh: makes the WW-only vs. WW+ZZ Fisher-matrix comparison in
# step 9 a direct matrix sum, no basis change needed.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"
MG5_DIR="$ROOT_DIR/MG5_aMC_v2_9_27"
MODEL="SMEFTsim_topU3l_MwScheme_UFO"
MODEL_DIR="$MG5_DIR/models/$MODEL"
RESTRICT_NAME="ww6op_massless"          # same restriction as create_ww.sh
OUT_RESTRICT="$MODEL_DIR/restrict_${RESTRICT_NAME}.dat"

if [ ! -f "$OUT_RESTRICT" ]; then
    echo "ERROR: $OUT_RESTRICT not found -- run create_ww.sh first" \
         "(it builds the shared 6-operator restriction card)." >&2
    exit 1
fi

PROC_DIR="$MG5_DIR/proc_cards"
mkdir -p "$PROC_DIR"
OUT_ZZ="$MG5_DIR/PROC_ZZ_emu_NP1"

cat > "$PROC_DIR/proc_card_zz_emu_NP1.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT_NAME
generate p p > e+ e- mu+ mu- NP=1
output $OUT_ZZ
EOF

if [ -d "$OUT_ZZ" ]; then
    echo "[skip] $OUT_ZZ already exists -> skipping output generation"
else
    "$MG5_DIR/bin/mg5_aMC" "$PROC_DIR/proc_card_zz_emu_NP1.dat"
fi

# --- same run_card tweaks as create_ww.sh/create_ccdy.sh: true dynamical
#     scale choice + reweighting systematics on ------------------------
RUNCARD="$OUT_ZZ/Cards/run_card.dat"
sed -E -i.bak \
    -e 's/^([[:space:]]*)-?[0-9]+([[:space:]]+= dynamical_scale_choice)/\1-1\2/' \
    -e 's/^([[:space:]]*)False([[:space:]]+= use_syst)/\1True\2/' \
    -e 's/^none([[:space:]]+= systematics_program)/systematics\1/' \
    "$RUNCARD"
rm -f "$RUNCARD.bak"

# --- reweight card: the same 34-point benchmark grid from reweight_points.py ---
mkdir -p "$OUT_ZZ/Cards"
python3 "$HERE/reweight_points.py" > "$OUT_ZZ/Cards/reweight_card.dat"
echo "==> Wrote $OUT_ZZ/Cards/reweight_card.dat ($(grep -c '^launch' "$OUT_ZZ/Cards/reweight_card.dat") points)"

# --- generate events, multicore --------------------------------------------
NEVENTS="${NEVENTS:-30000}"
NCORES="${NCORES:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

if [ -d "$OUT_ZZ/Events/run_01" ]; then
    echo "[skip] $OUT_ZZ/Events/run_01 already exists -> skipping event generation"
else
    sed -E -i.bak "s/^([[:space:]]*)[0-9]+([[:space:]]+= nevents)/\1${NEVENTS}\2/" "$RUNCARD"
    rm -f "$RUNCARD.bak"
    (cd "$OUT_ZZ" && echo "$PWD" && ./bin/generate_events run_01 -f --multicore --nb_core="$NCORES")
fi

# --- reweight (serial -- see create_ww.sh's comment: -R is broken for
#     generate_events in MG5 v2.9.27, and madevent reweight force_run=True
#     ignores --multicore/--nb_core anyway) -----------------------------
if [ -f "$OUT_ZZ/Events/run_01/unweighted_events.lhe.gz" ] && \
   zgrep -q "cll1_m1" "$OUT_ZZ/Events/run_01/unweighted_events.lhe.gz" 2>/dev/null; then
    echo "[skip] $OUT_ZZ/Events/run_01 already reweighted -> skipping reweight"
else
    (cd "$OUT_ZZ" && ./bin/madevent reweight run_01 -from_cards)
fi

echo "==> Done. LHE file: $OUT_ZZ/Events/run_01/unweighted_events.lhe.gz"
