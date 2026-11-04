#!/usr/bin/env python3
"""Run the source/config audit in the current evidence directory, preserving history."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT), str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import audit_current_profile_configs as audit
    audit.OUT = args.output.resolve()
    audit.OUT.mkdir(parents=True, exist_ok=True)
    audit.main()
    path = audit.OUT/'config_audit.json'
    report = json.loads(path.read_text())
    report['source_audit_harness_sha256'] = hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    report['template_diffs_note'] = 'No historical before_configs copied into this new directory; added flags are inventory-relative, not newly authored recipes.'
    report['profile_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'configs/paper_recipes').glob('*.json'))}
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
