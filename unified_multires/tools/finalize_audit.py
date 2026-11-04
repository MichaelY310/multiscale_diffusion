#!/usr/bin/env python3
"""Record environment, pinned source integrity, deliverable links and own GPU jobs."""
import datetime,hashlib,importlib.metadata,json,os,platform,socket,subprocess,sys
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
EVIDENCE=ROOT/'evidence'
def write(name,value):(EVIDENCE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    packages={}
    names=['torch','torchvision','numpy','scipy','Pillow','diffusers','transformers','timm','einops','accelerate','safetensors','opencv-python','opencv-python-headless','torchdiffeq','PyWavelets','ml_collections','beartype','simple_parsing','docstring_parser','pytest','omegaconf','imageio','tqdm','huggingface-hub','requests','pandas','packaging','ftfy']
    for name in names:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    import torch
    write('ENVIRONMENT.json',dict(audited_at_utc=now,hostname=socket.gethostname(),python=sys.version,executable=sys.executable,platform=platform.platform(),torch_cuda_build=torch.version.cuda,execution_device='cpu',gpu_used=False,isolated_dependencies=str(ROOT/'_deps'),packages=packages))
    req=['# Observed tested runtime, not an upstream paper environment lock.','# Existing FastVideo venv was used read-only; isolated additions live in _deps.','# Torch CUDA build is recorded in ENVIRONMENT.json; all validation was CPU.']
    req.extend(f'{name}=={version.split("+")[0]}' for name,version in packages.items())
    (ROOT/'requirements-authors.txt').write_text('\n'.join(req)+'\n')
    sources=json.loads((EVIDENCE/'sources.json').read_text());records={};bad=[];licenses=[]
    for key,source in sources.items():
        directory=ROOT/'sources'/key;changed=[]
        for relative,expected in source['selected_files'].items():
            path=directory/relative
            if not path.exists() or sha(path)!=expected:changed.append(relative)
        result=subprocess.run(['git','-C',str(directory),'rev-parse','HEAD'],capture_output=True,text=True)
        head=result.stdout.strip();pin_matches=head==source['commit']
        records[key]=dict(files=len(source['selected_files']),changed_files=changed,pinned_commit=source['commit'],head_commit=head,pin_matches=pin_matches)
        if changed or not pin_matches:bad.append(key)
        files=[p for p in source['selected_files'] if any(t in Path(p).name.lower() for t in ('license','licence','copying','notice'))]
        links='; '.join(f'[{p}](sources/{key}/{p})' for p in files) or 'No standalone license file in the selected release; see repository metadata/README.'
        licenses.append(f'|{key}|[{source["repo"].split("github.com/")[-1]}]({source["repo"]})|`{source["commit"]}`|{links}|')
    text=['# 源码、版本与许可证','','本框架使用固定 commit 的作者网络类与明确标注的求解器 backend。算法对照通过原始函数体/模块执行；作者文件未改写。各项目适用各自许可证，未将作者源码统一改称本框架许可。','','|来源|仓库|固定 commit|许可证文件|','|---|---|---|---|',*licenses,'','RDD 使用迁移后的 `multiscale_diffusion` 本地源码；其源文件 hash 见各 RDD oracle 证据。RealESRGAN 是 MrFlow 引用的确切 gitlink。RMD 仓库自称官方、metadata 作者名匹配，但论文页未链接它，来源归属未独立确认。','','预审记录：`evidence/PREIMPLEMENTATION_REVIEW.json`、`REALESRGAN_PREIMPLEMENTATION_REVIEW.json`、`RMD_PREIMPLEMENTATION_REVIEW.json`。早期 source-only 获取中额外 manifest/media 获取记录在 `evidence/acquisition_incident.json`；获取逻辑已修复。不包含模型或训练数据下载。']
    (ROOT/'SOURCES_AND_LICENSES.md').write_text('\n'.join(text)+'\n')
    artifact_records=[];artifact_failures=[]
    for pattern in ('*_conformance.json','*_training.json','*_inference.json'):
        for report in sorted(EVIDENCE.glob(pattern)):
            document=json.loads(report.read_text())
            for check in document.get('checks',[]):
                if 'artifact' not in check:continue
                artifact=check['artifact'];path=ROOT/artifact['tensor_pair']
                pair=torch.load(path,map_location='cpu',weights_only=True)
                same=torch.equal(pair['native'],pair['unified'])
                error=(pair['native'].double()-pair['unified'].double()).abs()
                maximum=error.max().item() if error.numel() else 0.
                valid=same==check['bitwise'] and maximum==check['max_abs'] and pair['name']==check['name']
                if not valid:artifact_failures.append(artifact['tensor_pair'])
                hashes={artifact['tensor_pair']:sha(path)}
                for label in ('native_png','unified_png','difference_png'):
                    if label in artifact:hashes[artifact[label]]=sha(ROOT/artifact[label])
                artifact_records.append(dict(report=str(report.relative_to(ROOT)),name=check['name'],bitwise=same,max_abs=maximum,valid=valid,sha256=hashes))
    write('SAMPLE_ARTIFACT_AUDIT.json',dict(audited_at_utc=now,status='passed' if not artifact_failures else 'failed',tensor_pairs=len(artifact_records),bitwise_pairs=sum(r['bitwise'] for r in artifact_records),failed=artifact_failures,artifacts=artifact_records))
    manifest={}
    for name in ('umdiff','tools','tests','examples','configs'):
        for path in sorted((ROOT/name).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:manifest[str(path.relative_to(ROOT))]=sha(path)
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ('.md','.html','.toml','.sh','.txt','.json'):manifest[path.name]=sha(path)
    missing=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and value and not value.startswith(('http:','https:','#','mailto:')):
                    target=ROOT/value.split('#')[0]
                    if target.parent==EVIDENCE and target.name in ('INTEGRITY_AUDIT.json','FINAL_PROCESS_AUDIT.json'):continue
                    if not target.exists():missing.append(value)
    report=ROOT/'REPRODUCTION_REPORT_ZH.html'
    if report.exists():Links().feed(report.read_text())
    write('INTEGRITY_AUDIT.json',dict(audited_at_utc=now,status='passed' if not bad and not missing else 'failed',sources=records,all_pinned_author_source_files_unchanged=not bad,missing_report_links=missing,deliverable_sha256=manifest))
    # Store only this task's process details. Other users/jobs are not disturbed.
    ours=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            cmd=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=os.readlink(directory/'cwd')
        except (OSError,PermissionError):continue
        if 'unified_multires' in cmd or cwd.startswith(str(ROOT)):
            ours.append(dict(pid=int(directory.name),command=cmd[:400],cwd=cwd))
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    query_exit=result.returncode;gpu_csv=result.stdout;snapshot_at=now;query_mode='current process'
    if query_exit and (EVIDENCE/'NVML_READONLY_SNAPSHOT.json').exists():
        snapshot=json.loads((EVIDENCE/'NVML_READONLY_SNAPSHOT.json').read_text())
        query_exit=snapshot['exit_code'];gpu_csv=snapshot['csv'];snapshot_at=snapshot['audited_at_utc'];query_mode=snapshot['execution']
    gpu_pids=set()
    if query_exit==0:
        for line in gpu_csv.splitlines():
            try:gpu_pids.add(int(line.split(',')[0].strip()))
            except ValueError:pass
    own_gpu=[r for r in ours if r['pid'] in gpu_pids]
    write('FINAL_PROCESS_AUDIT.json',dict(audited_at_utc=now,hostname=socket.gethostname(),gpu_ever_used=False,nvidia_smi_exit_code=query_exit,sandbox_nvml_exit_code=result.returncode,gpu_snapshot_at=snapshot_at,gpu_query_mode=query_mode,total_gpu_compute_processes=len(gpu_pids),own_gpu_processes=own_gpu,own_gpu_process_count=len(own_gpu),authorized_gpu=7,authorized_deadline_utc='2026-10-08T16:00:00+00:00',no_long_training_or_old_monitor_started=True,shared_dirty_fastvideo_modified=False,status='passed' if query_exit==0 and not own_gpu else 'needs_review'))
    print(json.dumps(dict(source_repositories=len(sources),source_files=sum(r['files'] for r in records.values()),source_unchanged=not bad,deliverable_files=len(manifest),missing_links=missing,own_gpu_processes=len(own_gpu),nvidia_smi_exit_code=query_exit)))
    if bad or missing or own_gpu or query_exit or artifact_failures:raise SystemExit('Final audit failed')
if __name__=='__main__':main()
