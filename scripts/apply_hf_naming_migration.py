#!/usr/bin/env python3
"""Hash-checked HF server-side rename. Token is read with getpass, never saved."""
import argparse
import getpass
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--check-auth',action='store_true')
parser.add_argument('--sync-registry',action='store_true')
args=parser.parse_args()
token=getpass.getpass('HF write credential (hidden): ')
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
import requests

def metadata(item):
    return dict(size=item.size,blob_id=item.blob_id,sha256=getattr(item.lfs,'sha256',None))

try:
    api=HfApi(token=token);who=api.whoami()
    print(json.dumps({'authenticated_user':who['name'],'token_role':who.get('auth',{}).get('accessToken',{}).get('role')}),flush=True)
    if args.check_auth:sys.exit(0)
    registry=ROOT/'docs/experiment_registry'
    plan=json.loads((registry/'hf_rename_plan.json').read_text());repo=plan['repo_id']
    info=api.model_info(repo,files_metadata=True)
    before={a.rfilename:metadata(a) for a in info.siblings}
    if args.sync_registry:
        if plan.get('status')!='applied_and_hash_verified':raise ValueError('Rename must be verified before metadata synchronization')
        for item in plan['operations']:
            assert before.get(item['new_path'])=={k:item[k] for k in ('size','blob_id','sha256')}
        updates=[CommitOperationAdd(path_in_repo='naming/'+name,path_or_fileobj=(registry/name).read_bytes())
                 for name in ('results.csv','results.json','models.json','configs.csv','repository_aliases.json','remote_fm_reports.json','hf_rename_plan.json')]
        commit=api.create_commit(repo,operations=updates,parent_commit=info.sha,commit_message='Synchronize reviewed RDD naming registry and compatibility map')
        receipt=ROOT/'docs/validation/20261104/HF_NAMING_MIGRATION.json'
        report=json.loads(receipt.read_text());report['registry_revision']=commit.oid;receipt.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'registry_revision':commit.oid,'model_hashes_rechecked':len(plan['operations'])}),flush=True)
        sys.exit(0)
    if info.sha!=plan['source_revision']:
        raise ValueError('Remote HEAD changed; regenerate/review the plan before mutation')
    operations=[]
    for item in plan['operations']:
        assert item['old_path'] in before,item['old_path']
        assert item['new_path'] not in before,item['new_path']
        assert before[item['old_path']]=={k:item[k] for k in ('size','blob_id','sha256')},item['old_path']
        operations.append(CommitOperationCopy(src_path_in_repo=item['old_path'],path_in_repo=item['new_path'],src_revision=info.sha))
        operations.append(CommitOperationDelete(path_in_repo=item['old_path']))
    print(json.dumps({'checked_moves':len(plan['operations']),'parent_commit':info.sha,'apply':args.apply}),flush=True)
    if not args.apply:sys.exit(0)
    # Preserve the original card, including its YAML frontmatter.
    if 'README.md' in before:
        response=requests.get(f'https://huggingface.co/{repo}/raw/{info.sha}/README.md',headers={'Authorization':'Bearer '+token},timeout=60)
        response.raise_for_status();card=response.text
    else:
        card=''
    note=('\n# RDD experiment names: resolution-labelled registry\n\n'
          'Current model folders spell out training duration at each latent grid (r8/r16/r32). '
          'Sampling budgets are separate: historical DDPM filenames saying 908080 generally ran r8=80, r16=80, r32=90; '
          'FM files saying 808090 use the same budget. DDPM training 300/300/400 and FM training 300/300/400 have different grid orders.\n\n'
          '- [All verified historical results and explicit grid allocations](naming/results.csv)\n'
          '- [Full result records with untouched source rows](naming/results.json)\n'
          '- [Model / training-clock registry](naming/models.json)\n'
          '- [Old path → current path and original content hashes](naming/hf_rename_plan.json)\n'
          '- [Configuration inventory; not a list of completed runs](naming/configs.csv)\n\n'
          'Weights and metric values are unchanged. Raw historical source files retain their original names; '
          'use the registry instead of inferring configuration from filenames. '
          f'Existing pinned downloads at revision `{info.sha}` remain valid. '
          'The archive below is historical provenance, not the current naming specification.\n\n---\n')
    if card.startswith('---'):
        end=card.find('\n---',3)
        assert end>=0
        card=card[:end+4]+note+card[end+4:]
    else:card=note+card
    operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()))
    published_plan=dict(plan,status='applied_in_this_commit')
    operations.append(CommitOperationAdd(path_in_repo='naming/hf_rename_plan.json',path_or_fileobj=json.dumps(published_plan,ensure_ascii=False,indent=2).encode()))
    for name in ('results.csv','results.json','models.json','configs.csv','repository_aliases.json'):
        operations.append(CommitOperationAdd(path_in_repo='naming/'+name,path_or_fileobj=(registry/name).read_bytes()))
    commit=api.create_commit(repo,operations=operations,parent_commit=info.sha,
        commit_message='Name RDD checkpoints and results by explicit latent-grid allocations',
        commit_description='Hash-preserving server-side renames; separate training clocks, sampling budgets and NFE. Original revision remains accessible.')
    after_info=api.model_info(repo,revision=commit.oid,files_metadata=True)
    after={a.rfilename:metadata(a) for a in after_info.siblings}
    checks=[]
    for item in plan['operations']:
        ok=after.get(item['new_path'])==before[item['old_path']] and item['old_path'] not in after
        checks.append({'old_path':item['old_path'],'new_path':item['new_path'],'content_identity_preserved':ok})
    report=dict(repo_id=repo,source_revision=info.sha,renamed_revision=commit.oid,url=commit.commit_url,
                checked=len(checks),all_passed=all(x['content_identity_preserved'] for x in checks),checks=checks)
    dest=ROOT/'docs/validation/20261104/HF_NAMING_MIGRATION.json';dest.write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_passed']:raise ValueError('Post-commit hash check failed; see receipt, no further changes made')
    plan.update(status='applied_and_hash_verified',applied_revision=commit.oid)
    (registry/'hf_rename_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    proof=api.create_commit(repo,operations=[CommitOperationAdd(path_in_repo='naming/migration_receipt.json',path_or_fileobj=dest.read_bytes())],parent_commit=commit.oid,commit_message='Record post-rename content identity verification')
    report['receipt_revision']=proof.oid;dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
except Exception as error:
    message=str(error).replace(token,'[redacted]')
    print(type(error).__name__+': '+message,file=sys.stderr)
    sys.exit(1)
finally:
    token=None
