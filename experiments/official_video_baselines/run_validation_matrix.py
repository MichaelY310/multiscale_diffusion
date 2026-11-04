"""Sequential fresh-process jobs within one independently guarded GPU lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('RDD_JOB_TOKEN'):
        raise RuntimeError('Use the independent GPU guard')
    jobs=json.loads(args.jobs.read_text()); results=[]
    for job in jobs:
        if time.time()+180>=float(os.environ['RDD_DEADLINE_UNIX']):
            raise RuntimeError('Not enough authorized time for another validation process')
        log=Path(job['log']);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            started=time.time()
            completed=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node='+str(job['ranks']),*job['arguments']],stdout=stream,stderr=subprocess.STDOUT)
        results.append({'job':job,'exit_code':completed.returncode,'elapsed_seconds':time.time()-started})
        args.output.write_text(json.dumps({'status':'running','results':results},indent=2))
        if completed.returncode:raise RuntimeError('Failed job; see '+str(log))
        print(json.dumps(results[-1]),flush=True)
    args.output.write_text(json.dumps({'status':'passed','results':results},indent=2))


if __name__=='__main__':main()
