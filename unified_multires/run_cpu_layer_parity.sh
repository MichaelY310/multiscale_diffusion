#!/usr/bin/env bash
# Full author-core CPU layer parity. Synthetic fixtures, never paper training.
set -euo pipefail
unified_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$unified_root/activate_cpu.sh"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python_bin=${RDD_PARITY_PYTHON:-$UMDIFF_PYTHON}
evidence_root=${1:-"$unified_root/evidence/forward_backward_20261012"}
threads=${RDD_PARITY_THREADS:-8}
mkdir -p "$evidence_root"
"$python_bin" "$unified_root/tools/record_cpu_parity_environment.py" --output "$evidence_root"
"$python_bin" "$unified_root/tools/complete_current_cpu_matrix.py" --output "$evidence_root/full_current" --steps 2 --threads "$threads"
"$python_bin" "$unified_root/tools/validate_full_forward_backward.py" --all --reduced --steps 2 --threads "$threads" --output "$evidence_root/reduced_release"
"$python_bin" "$unified_root/tools/validate_activations.py" --all --steps 128 --workers 2 --directory "$evidence_root/long_release"
for method in dctdiff mdm mdm_vdm ssd ppflow dimr dfm lapflow lapflow3 relay rdd diffusability; do
  "$python_bin" "$unified_root/tools/validate_recipe_extensions.py" "$method" --steps 128 --threads 2 --directory "$evidence_root/extensions_current/$method"
done
"$python_bin" "$unified_root/tools/validate_recipe_extensions.py" diffusability --full-codec --steps 2 --threads "$threads" --directory "$evidence_root/codec_current"
"$python_bin" "$unified_root/tools/validate_rmd_current_profile.py" --full-backbone --steps 17 --threads "$threads" --output "$evidence_root/rmd"
"$python_bin" "$unified_root/tools/validate_ssd_resolution_branches.py" --threads "$threads" --output "$evidence_root/ssd_resolution_branches.json"
for method in speed nami mrflow; do
  "$python_bin" "$unified_root/tools/validate_inference.py" "$method" --output-dir "$evidence_root/inference_layers"
done
"$python_bin" "$unified_root/tools/validate_cat_published_codec.py" --steps 2 --threads 4 --output "$evidence_root/cat_codec"
"$python_bin" "$unified_root/tools/validate_cat_published_codec.py" --full-backbone --steps 1 --threads "$threads" --output "$evidence_root/cat_codec_full"
"$python_bin" "$unified_root/tools/validate_cat_api_contract.py" --output "$evidence_root/cat_api"
"$python_bin" "$unified_root/tools/validate_cpu_gap_fixes.py" --output "$evidence_root/gap_regressions.json"
for method in relay ppflow rdd lapflow; do
  "$python_bin" "$unified_root/tools/validate_generic_components_current.py" "$method" --output "$evidence_root/components/$method.json"
done
"$python_bin" "$unified_root/tools/verify_cpu_checkpoint_states.py" --evidence "$evidence_root"
"$python_bin" "$unified_root/tools/build_cpu_completion_report.py" --evidence "$evidence_root"
"$python_bin" "$unified_root/tools/cleanup_cpu_parity_payloads.py" --evidence "$evidence_root" --apply
"$python_bin" "$unified_root/tools/update_cpu_completion_handoffs.py" --evidence "$evidence_root"
