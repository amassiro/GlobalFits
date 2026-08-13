#!/usr/bin/env bash
set -euo pipefail

# create_ccdy.sh -- generate the "closing process" sample for step 7-9 of
# the flat-directions notebook: charged-current Drell-Yan, p p > e+ ve
# (+ charge conjugate), NP=1, same 6-operator restriction/benchmark grid as
# create_ww.sh (see that script + reweight_points.py for the full
# rationale). Reusing the identical 6D basis here is deliberate: it's what
# makes the WW-only vs. WW+CCDY Fisher-matrix comparison in step 9 a direct,
# apples-to-apples matrix sum instead of needing a basis change.
#
# Electron channel only: in this UFO's flavour scheme (SMEFTsim_topU3l_
# MwScheme -- U(3)^5-symmetric leptons except top handled separately on the
# quark side), cHl3/cll1/cHj3 are lepton-flavour-universal by construction
# (a single parameter each, no e/mu/tau index in parameters.py) -- so the
# muon channel would give numerically identical sensitivity and adds
# nothing. Left as an easy exercise to check this empirically if you want.
#
# This is a genuine 2->2 process (unlike WW's 2->4), so it's much cheaper:
# default statistics are set higher accordingly.

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
OUT_CCDY="$MG5_DIR/PROC_CCDY_e_NP1"

cat > "$PROC_DIR/proc_card_ccdy_e_NP1.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT_NAME
generate p p > e+ ve NP=1
add process p p > e- ve~ NP=1
output $OUT_CCDY
EOF

if [ -d "$OUT_CCDY" ]; then
    echo "[skip] $OUT_CCDY already exists -> skipping output generation"
else
    "$MG5_DIR/bin/mg5_aMC" "$PROC_DIR/proc_card_ccdy_e_NP1.dat"
fi

RUNCARD="$OUT_CCDY/Cards/run_card.dat"
sed -E -i.bak \
    -e 's/^([[:space:]]*)-?[0-9]+([[:space:]]+= dynamical_scale_choice)/\1-1\2/' \
    -e 's/^([[:space:]]*)False([[:space:]]+= use_syst)/\1True\2/' \
    -e 's/^none([[:space:]]+= systematics_program)/systematics\1/' \
    "$RUNCARD"
rm -f "$RUNCARD.bak"

mkdir -p "$OUT_CCDY/Cards"
python3 "$HERE/reweight_points.py" > "$OUT_CCDY/Cards/reweight_card.dat"
echo "==> Wrote $OUT_CCDY/Cards/reweight_card.dat ($(grep -c '^launch' "$OUT_CCDY/Cards/reweight_card.dat") points)"

NEVENTS="${NEVENTS:-60000}"
NCORES="${NCORES:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

if [ -d "$OUT_CCDY/Events/run_01" ]; then
    echo "[skip] $OUT_CCDY/Events/run_01 already exists -> skipping event generation"
else
    sed -E -i.bak "s/^([[:space:]]*)[0-9]+([[:space:]]+= nevents)/\1${NEVENTS}\2/" "$RUNCARD"
    rm -f "$RUNCARD.bak"
    (cd "$OUT_CCDY" && echo "$PWD" && ./bin/generate_events run_01 -f --multicore --nb_core="$NCORES")
fi

if [ -f "$OUT_CCDY/Events/run_01/unweighted_events.lhe.gz" ] && \
   zgrep -q "cll1_m1" "$OUT_CCDY/Events/run_01/unweighted_events.lhe.gz" 2>/dev/null; then
    echo "[skip] $OUT_CCDY/Events/run_01 already reweighted -> skipping reweight"
else
    (cd "$OUT_CCDY" && ./bin/madevent reweight run_01 -from_cards)
fi

echo "==> Done. LHE file: $OUT_CCDY/Events/run_01/unweighted_events.lhe.gz"
