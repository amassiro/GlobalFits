#!/usr/bin/env bash
set -euo pipefail

# --- config ---------------------------------------------------------------
MG5_DIR="$PWD/MG5_aMC_v2_9_27"
MODEL="SMEFTsim_topU3l_MwScheme_UFO"
RESTRICT="cHWB_massless"          # restrict_cHWB_massless.dat: every operator fixed=0
                                   # except cHWB, set to a generic non-special placeholder
                                   # (9.999999e-01, not exactly 1) -- since that value isn't
                                   # "special" (0 or 1), MG5 keeps cHWB as an external/settable
                                   # Block SMEFT entry in param_card.dat rather than hardcoding
                                   # it away. That's what lets this one card serve both roles
                                   # below: the ~1 benchmark for the SM/linear/quadratic
                                   # decomposition (sigma(c) = sigma_SM + c*sigma_lin +
                                   # c^2*sigma_quad), AND PROC_cHWB_NP1's reweight_card, which
                                   # needs cHWB to stay externally settable.

PROC_DIR="$MG5_DIR/proc_cards"
mkdir -p "$PROC_DIR"

OUT_NP1="$MG5_DIR/PROC_cHWB_NP1"           # 1) NP=1, cHWB on
OUT_SM="$MG5_DIR/PROC_SM_NP0"               # 2) NP=0, pure SM
OUT_LIN="$MG5_DIR/PROC_cHWB_linear"         # 3) NP=1 NP^2==1, interference
OUT_QUAD="$MG5_DIR/PROC_cHWB_quadratic"     # 4) NP=1 NP^2==2, pure EFT^2

# 1) NP=1, cHWB on -> full SM + interference + EFT^2 sample
#    (this is the sample that gets reweighted over cHWB by reweight_card.dat below --
#    see the RESTRICT comment above for why cHWB stays settable under this same card)
cat > "$PROC_DIR/proc_card_1_NP1_cHWB.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT
generate p p > l+ l- NP=1
output $OUT_NP1
EOF

# 2) NP=0 -> pure SM
cat > "$PROC_DIR/proc_card_2_SM.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT
generate p p > l+ l- NP=0
output $OUT_SM
EOF

# 3) NP=1 NP^2==1 -> linear (SM x EFT interference) only
cat > "$PROC_DIR/proc_card_3_linear.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT
generate p p > l+ l- NP=1 NP^2==1
output $OUT_LIN
EOF

# 4) NP=1 NP^2==2 -> quadratic (pure EFT^2) only
cat > "$PROC_DIR/proc_card_4_quadratic.dat" <<EOF
set auto_convert_model T
import model $MODEL-$RESTRICT
generate p p > l+ l- NP=1 NP^2==2
output $OUT_QUAD
EOF

# --- run MG5 to actually produce the 4 output directories -----------------
run_proc_card() {
    local pc="$1"
    local out_dir="$2"
    if [ -d "$out_dir" ]; then
        echo "[skip] $out_dir already exists -> skipping output generation for $pc"
    else
        "$MG5_DIR/bin/mg5_aMC" "$PROC_DIR/${pc}.dat"
    fi
}

run_proc_card proc_card_1_NP1_cHWB   "$OUT_NP1"
run_proc_card proc_card_2_SM         "$OUT_SM"
run_proc_card proc_card_3_linear     "$OUT_LIN"
run_proc_card proc_card_4_quadratic  "$OUT_QUAD"

# --- reweight card for option 1: cHWB = 0, +1, -1 (Block SMEFT, idx 9) ----
mkdir -p "$OUT_NP1/Cards"
cat > "$OUT_NP1/Cards/reweight_card.dat" <<EOF
launch --rwgt_name=cHWB_0
  set SMEFT 9 0.0

launch --rwgt_name=cHWB_p1
  set SMEFT 9 1.0

launch --rwgt_name=cHWB_m1
  set SMEFT 9 -1.0
EOF





# --- Run events ---

# --- generate events, multicore --------------------------------------------
NCORES="${NCORES:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"   # cores per run, auto-detected

for OUT in "$OUT_NP1" "$OUT_SM" "$OUT_LIN" "$OUT_QUAD"; do
    if [ -d "$OUT/Events/run_01" ]; then
        echo "[skip] $OUT/Events/run_01 already exists -> skipping event generation"
        continue
    fi
    # NOTE: don't pass -R here. In this MG5 version (v2.9.27) generate_events
    # fails to strip -R before forwarding leftover args into the internal
    # `survey` call, which crashes with "Too many argument for survey command".
    # Reweighting is run as its own step below instead.
    (cd "$OUT" && echo $PWD && ./bin/generate_events run_01 -f --multicore --nb_core="$NCORES")
done

# --- reweight PROC_cHWB_NP1 separately (works around the -R bug above) -----
# bin/madevent always runs with force_run=True, so --multicore/--nb_core
# wouldn't do anything for reweight anyway -> run it serially, it's cheap
# compared to survey/refine.
if [ -f "$OUT_NP1/Events/run_01/unweighted_events.lhe.gz" ] && \
   zgrep -q "cHWB_0" "$OUT_NP1/Events/run_01/unweighted_events.lhe.gz" 2>/dev/null; then
    echo "[skip] $OUT_NP1/Events/run_01 already reweighted -> skipping reweight"
else
    (cd "$OUT_NP1" && ./bin/madevent reweight run_01 -from_cards)
fi
