"""Resolve renamed RDD archive paths without breaking historical callers.

New calls should use canonical paths. A legacy path selects its immutable
pre-rename revision; this also preserves the old local_dir layout for scripts
which explicitly require that path. Other repositories are unaffected.
"""
import json
from pathlib import Path

def archive_revision(repo_id, filename, revision=None):
    if revision is not None or repo_id!='yang335/search_agent_hl_to_i18n_transport':
        return revision
    plan=json.loads((Path(__file__).resolve().parents[1]/'docs/experiment_registry/hf_rename_plan.json').read_text())
    if any(r['old_path']==filename for r in plan['operations']):
        print('Legacy archive path: using immutable revision '+plan['source_revision']+'. See docs/RDD_EXPERIMENT_NAMES_ZH.md for the canonical name.')
        return plan['source_revision']
    return revision
