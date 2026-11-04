"""Re-label the existing 116-row result ledger without changing original records.

This is a source/metadata audit, not a rerun or a per-run executable hash proof.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent
OUT = ROOT / 'evidence/rdd_sampling_scope_20261104'
LEDGER = OLD / 'eval_results/experiment_ledger_20260918/curated_valid_results.csv'


def main():
    rows = list(csv.DictReader(LEDGER.open(encoding='utf-8-sig', newline='')))
    for row in rows:
        if row['method'] == 'RDD diffusion' and row['transition'] == 'noise filling':
            label, detail = 'fresh', 'nearest clean upsampling + full independent Gaussian'
            evidence = 'Archived transition_ablation README explicitly identifies historical 2-D noise_filling as redraw; current DDPM dispatcher agrees.'
        elif row['method'] == 'RDD diffusion' and row['transition'] == 'bilinear':
            label, detail = 'fresh', 'bilinear clean upsampling + full independent Gaussian'
            evidence = 'Current _bilinear_upsample consumes clean, target time and new noise only; historical ledger marks bilinear.'
        elif row['method'] == 'RDD flow matching' and row['transition'] == 'noise filling':
            label, detail = 'noise_filling', 'recover noise from endpoint noisy state and estimated clean, normalize lift, add null-space noise'
            evidence = 'Saved FM evaluation launchers select fm_unified/noise_filling; sample_rdd_fm_multistage dispatches noise_filling_transition_fm.'
        elif row['method'] == 'Original diffusion' and row['transition'] == 'n.a.':
            label, detail = 'not_applicable', 'single-resolution DiT, no resolution boundary'
            evidence = 'Existing ledger method and topology.'
        else:
            raise ValueError(f'Unreviewed ledger category: {row}')
        row.update(boundary_definition_20261104=label, boundary_detail_20261104=detail,
                   classification_evidence_20261104=evidence,
                   classification_scope_20261104='Existing reports, launchers and retained implementation; no formal rerun or pinned per-run executable snapshot certification.')
    counts = dict(collections.Counter(r['boundary_definition_20261104'] for r in rows))
    assert counts == {'fresh': 98, 'not_applicable': 13, 'noise_filling': 5}, counts
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'HISTORICAL_IMAGE_RESULTS_RELABELED.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sources = [LEDGER, OLD/'eval_results/transition_ablation/unified_l_300300400_400k/README.md',
               OLD/'2d/exp/sample_unified_l_transition_redraw_vs_noise_filling.sh',
               OLD/'2d/exp/eval_rdd_fm_unified_xl_v2_400k_2gpu.sh',
               OLD/'2d/exp/eval_rdd_fm_unified_l_convergence_4gpu.sh',
               OLD/'2d/sample_fid_unified.py', OLD/'diffusion/flow_matching_with_drift.py',
               OLD/'rdd_video/sample_fastwan_native_vsa.py',
               OLD/'rdd_video/diagnose_csbr_residual_hypotheses.py',
               OLD/'rdd_video/flow_matching_video.py',
               ROOT.parent/'FastVideo-RDD-minimal/rdd/sampler.py', Path(__file__)]
    record = dict(counts=counts, fresh_nearest=94, fresh_bilinear=4,
                  history_search='Searched local .codex/history.jsonl and 155 local session JSONL files for noise filling / redraw / innovation_preserving and renaming. No original renaming conversation recovered; only later messages, carried context and repository records. This is not a search of inaccessible cloud/other-machine conversations.',
                  excluded='252 historical wrong-data FID rows have no boundary field; not reclassified. Early video root_cause_f29 noise_filling tags have ambiguous historical semantics and are not certified by current dispatch alone.',
                  ddpm_preserve_exception='The DDPM preserve branch carries stage_noise initialized at the start of the previous stage; it does not recover residual noise from the stage endpoint. Keep separate from the new strict noise_filling definition.',
                  source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'HISTORICAL_CLASSIFICATION.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record['counts']))


if __name__ == '__main__':
    main()
