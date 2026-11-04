"""CPU coordinator; each public command obtains its own independent GPU guard."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for job in json.loads(args.jobs.read_text()):
        started = time.time()
        with Path(job['log']).open('x') as stream:
            process = subprocess.run(job['command'], stdout=stream, stderr=subprocess.STDOUT)
        results.append(dict(job=job, exit_code=process.returncode, elapsed_seconds=time.time() - started))
        args.output.write_text(json.dumps(dict(status='running', results=results), indent=2))
        if process.returncode:
            raise RuntimeError('Failed public entry; see ' + job['log'])
    args.output.write_text(json.dumps(dict(status='passed', results=results), indent=2))


if __name__ == '__main__':
    main()
