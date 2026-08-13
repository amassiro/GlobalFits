#!/usr/bin/env bash
set -euo pipefail

# create_vbs.sh -- generate the same-sign WW VBS SMEFT sample used by the
# "Wilks' theorem" hands-on exercise.
#
# Process: p p > e+ mu+ ve vm j j QCD=0 SMHLOOP=0 NP=1
#   - e+ mu+ ve vm: same-sign W+W+ -> e+ nu_e mu+ nu_mu decay chain, treated
#     as a genuine off-shell 2->6 leptonic+jets final state (same philosophy
#     as ../flat_directions/create_ww.sh: NP=<order> EFT power counting
#     handles production+decay vertices together automatically, no
#     narrow-width factorization ambiguity).
#   - QCD=0: pure electroweak production only -- excludes the QCD-induced
#     same-sign-WW+jets background (an irreducible SM process at QCD=2 that
#     is NOT part of "VBS" in the usual sense; this is the standard
#     amplitude-level trick used by real ATLAS/CMS VBS EFT analyses).
#   - SMHLOOP=0: excludes loop-induced SM contributions (real coupling
#     order defined in this UFO's coupling_orders.py -- verified directly
#     against the model files before writing this script, not assumed).
#   - NP=1: exactly matches flat_directions/create_ww.sh's convention --
#     |M_SM + M_NP|^2 gives SM (NP^0) + interference (NP^1) + EFT^2 (NP^2)
#     terms in the cross section automatically, which is exactly the exact
#     quadratic response surface fitlib.fit_quadratic_surface() solves for.
#
# Deliberately only the e+ mu+ (W+W+) charge combination, not also e- mu-
# (W-W-) -- matches the process line as specified for this exercise. Add a
# second `add process p p > e- mu- ve~ vm~ j j QCD=0 SMHLOOP=0 NP=1` line
# below (mirroring create_ww.sh's charge-conjugate handling) if more
# statistics are needed later; this does not change any physics conclusion,
# just doubles raw event count.
#
# Reuses this repo's already-installed MG5_aMC_v2_9_27 + SMEFTsim_topU3l_
# MwScheme_UFO, AND the exact same restrict_ww6op_massless.dat restriction
# card that ../flat_directions/create_ww.sh already created and validated
# (6 operators kept external; this exercise's reweight_points.py just never
# populates one of them, cll1, in any benchmark point -- see its docstring).
# Run ../setup.sh first if MG5_aMC_v2_9_27 doesn't exist yet.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"          # SPLIT_SCHOOL/MASSIRO
MG5_DIR="$ROOT_DIR/MG5_aMC_v2_9_27"
MODEL="SMEFTsim_topU3l_MwScheme_UFO"
MODEL_DIR="$MG5_DIR/models/$MODEL"
RESTRICT_NAME="ww6op_massless"
OUT_RESTRICT="$MODEL_DIR/restrict_${RESTRICT_NAME}.dat"

if [ ! -d "$MG5_DIR" ]; then
    echo "ERROR: $MG5_DIR not found -- run $ROOT_DIR/setup.sh first." >&2
    exit 1
fi
if [ ! -f "$OUT_RESTRICT" ]; then
    echo "ERROR: $OUT_RESTRICT not found -- run ../flat_directions/create_ww.sh" >&2
    echo "       first (or once) to create it; this script reuses it as-is." >&2
    exit 1
fi
echo "[reuse] $OUT_RESTRICT (already created by ../flat_directions/create_ww.sh)"

# --- proc card -------------------------------------------------------------
PROC_DIR="$MG5_DIR/proc_cards"
mkdir -p "$PROC_DIR"
OUT_VBS="$MG5_DIR/PROC_VBS_ssWW_emu_NP1"

cat > "$PROC_DIR/proc_card_vbs_ssWW_emu_NP1.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT_NAME
generate p p > e+ mu+ ve vm j j QCD=0 SMHLOOP=0 NP=1
output $OUT_VBS
EOF

if [ -d "$OUT_VBS" ]; then
    echo "[skip] $OUT_VBS already exists -> skipping output generation"
else
    "$MG5_DIR/bin/mg5_aMC" "$PROC_DIR/proc_card_vbs_ssWW_emu_NP1.dat"
fi

# --- force dynamical_scale_choice=-1 (MG5's true dynamical scale) and
#     enable use_syst, exactly as ../flat_directions/create_ww.sh does. -----
RUNCARD="$OUT_VBS/Cards/run_card.dat"
sed -E -i.bak \
    -e 's/^([[:space:]]*)-?[0-9]+([[:space:]]+= dynamical_scale_choice)/\1-1\2/' \
    -e 's/^([[:space:]]*)False([[:space:]]+= use_syst)/\1True\2/' \
    -e 's/^none([[:space:]]+= systematics_program)/systematics\1/' \
    "$RUNCARD"
rm -f "$RUNCARD.bak"

# --- reweight card: the 27-point benchmark grid (21 fit + 6 validation)
#     from this directory's reweight_points.py (5 operators) -------------
mkdir -p "$OUT_VBS/Cards"
python3 "$HERE/reweight_points.py" > "$OUT_VBS/Cards/reweight_card.dat"
echo "==> Wrote $OUT_VBS/Cards/reweight_card.dat ($(grep -c '^launch' "$OUT_VBS/Cards/reweight_card.dat") points)"

# --- generate events, multicore --------------------------------------------
NEVENTS="${NEVENTS:-30000}"
NCORES="${NCORES:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

if [ -d "$OUT_VBS/Events/run_01" ]; then
    echo "[skip] $OUT_VBS/Events/run_01 already exists -> skipping event generation"
else
    sed -E -i.bak "s/^([[:space:]]*)[0-9]+([[:space:]]+= nevents)/\1${NEVENTS}\2/" "$RUNCARD"
    rm -f "$RUNCARD.bak"
    (cd "$OUT_VBS" && echo "$PWD" && ./bin/generate_events run_01 -f --multicore --nb_core="$NCORES")
fi

# --- reweight (serial -- see ../flat_directions/create_ww.sh's comment:
#     -R is broken for generate_events in MG5 v2.9.27, and madevent reweight
#     force_run=True ignores --multicore/--nb_core anyway) -----------------
if [ -f "$OUT_VBS/Events/run_01/unweighted_events.lhe.gz" ] && \
   zgrep -q "cHl3_m1" "$OUT_VBS/Events/run_01/unweighted_events.lhe.gz" 2>/dev/null; then
    echo "[skip] $OUT_VBS/Events/run_01 already reweighted -> skipping reweight"
else
    (cd "$OUT_VBS" && ./bin/madevent reweight run_01 -from_cards)
fi

echo "==> Done. LHE file: $OUT_VBS/Events/run_01/unweighted_events.lhe.gz"
