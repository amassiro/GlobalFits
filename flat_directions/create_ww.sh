#!/usr/bin/env bash
set -euo pipefail

# create_ww.sh -- generate the WW -> e mu + MET (different-flavour, both
# charge assignments) SMEFT sample used by the flat-directions notebook.
#
# Process: p p > e+ ve mu- vm~  (+ charge conjugate e- ve~ mu+ vm), NP=1.
# This is the genuine off-shell 2->4 leptonic EW final state, NOT a narrow-
# width decay-chain factorization -- deliberately, for two reasons:
#   1) it sidesteps any ambiguity about how MadGraph's NP=<order> EFT
#      power-counting propagates through decay-chain syntax (production and
#      decay vertices are just counted together automatically here);
#   2) it's the more complete/realistic treatment anyway (matches how real
#      ATLAS/CMS off-shell WW SMEFT analyses treat it), and for this exact
#      flavour/charge combination there is no non-WW SM tree topology to
#      worry about missing: "e+ ve" / "mu- vm~" only arise from a W+/W-
#      current, so every SM+SMEFT tree diagram contributing here already IS
#      a (resonant or non-resonant) double-W-exchange diagram.
#
# Reuses this repo's already-installed MG5_aMC_v2_9_27 + SMEFTsim_topU3l_
# MwScheme_UFO (see ../setup.sh) -- run that first if MG5_aMC_v2_9_27
# doesn't exist yet as a sibling of this directory's parent.
#
# We do NOT re-validate reweighting against direct generation here: the
# sibling cHWB pipeline (../create.sh, ../analysis.py) already did that for
# this exact model to <1% precision (see ../analysis_output/mll_comparison.
# png) -- direct-generating all 34 benchmark points below instead of
# reweighting a single sample to them would be ~34x the MG5 integration
# cost for no new information.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"          # SPLIT_SCHOOL/MASSIRO
MG5_DIR="$ROOT_DIR/MG5_aMC_v2_9_27"
MODEL="SMEFTsim_topU3l_MwScheme_UFO"
MODEL_DIR="$MG5_DIR/models/$MODEL"

if [ ! -d "$MG5_DIR" ]; then
    echo "ERROR: $MG5_DIR not found -- run $ROOT_DIR/setup.sh first." >&2
    exit 1
fi

# --- restriction card: 6 operators live (external/settable), everything
#     else pruned to exactly 0. Same trick as ../setup.sh's cHWB card: a
#     restricted value of exactly 0 or 1 gets hardcoded away by MG5; any
#     other value (9.999999e-01 here, for all 6) keeps the parameter
#     external so reweight_card.dat can still set it later. -------------
OPS="cW cHWB cHW cHj3 cHl3 cll1"
VALUE="9.999999e-01"
BASE="$MODEL_DIR/restrict_SMlimit_massless.dat"
RESTRICT_NAME="ww6op_massless"
OUT_RESTRICT="$MODEL_DIR/restrict_${RESTRICT_NAME}.dat"

if [ -f "$OUT_RESTRICT" ]; then
    echo "[skip] $OUT_RESTRICT already exists"
else
    cp "$BASE" "$OUT_RESTRICT.tmp"
    for OP in $OPS; do
        sed -E -i.bak "s/^(([[:space:]]*[0-9]+)[[:space:]]+)0\.000000e\+00([[:space:]]*# ${OP} )\$/\\1${VALUE}\\3/" \
            "$OUT_RESTRICT.tmp"
        rm -f "$OUT_RESTRICT.tmp.bak"
    done
    mv "$OUT_RESTRICT.tmp" "$OUT_RESTRICT"
    echo "==> Wrote $OUT_RESTRICT"
    for OP in $OPS; do
        grep -a "# ${OP} \$" "$OUT_RESTRICT" | sed "s/^/    /"
    done
fi

# --- proc card -------------------------------------------------------------
PROC_DIR="$MG5_DIR/proc_cards"
mkdir -p "$PROC_DIR"
OUT_WW="$MG5_DIR/PROC_WW_emu_NP1"

cat > "$PROC_DIR/proc_card_ww_emu_NP1.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT_NAME
generate p p > e+ ve mu- vm~ NP=1
add process p p > e- ve~ mu+ vm NP=1
output $OUT_WW
EOF

if [ -d "$OUT_WW" ]; then
    echo "[skip] $OUT_WW already exists -> skipping output generation"
else
    "$MG5_DIR/bin/mg5_aMC" "$PROC_DIR/proc_card_ww_emu_NP1.dat"
fi

# --- force dynamical_scale_choice=-1 (MG5's true dynamical scale) and
#     enable use_syst, exactly as ../create.sh does. This process (plain
#     NP=1, no '^2' in the definition) is NOT auto-detected as
#     "interference" by MG5, so unlike ../create.sh's Lin/Quad samples this
#     is not fighting any auto-override -- it's just making the choice
#     explicit/reproducible. Note: the scale/PDF systematics weights this
#     produces are NOT consumed anywhere in fitlib.py -- read_yields() only
#     ever looks up the named reweight-benchmark weights (reweight_points.py's
#     grid). We still generate them (cheap relative to event generation
#     itself, and keeps this sample self-consistent with ../create.sh's
#     convention) but the notebook's likelihood/chi2 is defined purely as a
#     function of the Wilson coefficients via the fitted quadratic surface
#     (see chi2_factory()), not systematic nuisance parameters. -----------
RUNCARD="$OUT_WW/Cards/run_card.dat"
sed -E -i.bak \
    -e 's/^([[:space:]]*)-?[0-9]+([[:space:]]+= dynamical_scale_choice)/\1-1\2/' \
    -e 's/^([[:space:]]*)False([[:space:]]+= use_syst)/\1True\2/' \
    -e 's/^none([[:space:]]+= systematics_program)/systematics\1/' \
    "$RUNCARD"
rm -f "$RUNCARD.bak"

# --- reweight card: the 34-point benchmark grid from reweight_points.py ---
mkdir -p "$OUT_WW/Cards"
python3 "$HERE/reweight_points.py" > "$OUT_WW/Cards/reweight_card.dat"
echo "==> Wrote $OUT_WW/Cards/reweight_card.dat ($(grep -c '^launch' "$OUT_WW/Cards/reweight_card.dat") points)"

# --- generate events, multicore --------------------------------------------
NEVENTS="${NEVENTS:-30000}"
NCORES="${NCORES:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

if [ -d "$OUT_WW/Events/run_01" ]; then
    echo "[skip] $OUT_WW/Events/run_01 already exists -> skipping event generation"
else
    sed -E -i.bak "s/^([[:space:]]*)[0-9]+([[:space:]]+= nevents)/\1${NEVENTS}\2/" "$RUNCARD"
    rm -f "$RUNCARD.bak"
    (cd "$OUT_WW" && echo "$PWD" && ./bin/generate_events run_01 -f --multicore --nb_core="$NCORES")
fi

# --- reweight (serial -- see ../create.sh's comment: -R is broken for
#     generate_events in MG5 v2.9.27, and madevent reweight force_run=True
#     ignores --multicore/--nb_core anyway) -----------------------------
if [ -f "$OUT_WW/Events/run_01/unweighted_events.lhe.gz" ] && \
   zgrep -q "cll1_m1" "$OUT_WW/Events/run_01/unweighted_events.lhe.gz" 2>/dev/null; then
    echo "[skip] $OUT_WW/Events/run_01 already reweighted -> skipping reweight"
else
    (cd "$OUT_WW" && ./bin/madevent reweight run_01 -from_cards)
fi

echo "==> Done. LHE file: $OUT_WW/Events/run_01/unweighted_events.lhe.gz"
