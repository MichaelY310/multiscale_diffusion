"""Stop ONLY this probe's same-UID, exact-output process using a pidfd."""
import argparse
import os
import signal
import select

p=argparse.ArgumentParser()
p.add_argument('--pid',type=int,required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
proc=f'/proc/{a.pid}'
fd=os.pidfd_open(a.pid)
try:
    if os.stat(proc).st_uid!=os.getuid():
        raise PermissionError('Different UID')
    with open(proc+'/cmdline','rb') as stream:
        argv=stream.read().decode().split('\0')
    if 'rdd_video/progressive_infra/benchmark_temporal_band_memory.py' not in argv:
        raise PermissionError('Not the exact owned benchmark script')
    if '--output' not in argv or argv[argv.index('--output')+1]!=a.output:
        raise PermissionError('Output token mismatch')
    signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
    ready,_,_=select.select([fd],[],[],10)
    if not ready:
        signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
    print(f'Stopped same-UID exact-output probe PID {a.pid}: {a.output}')
finally:
    os.close(fd)
