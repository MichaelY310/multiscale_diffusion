"""Fixed-index scientific contact sheets and a portable result integrity index."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
import imageio_ffmpeg


def main(root,include_progress=False):
    root=Path(root)
    records=list(root.glob('*/metrics.json'))
    if include_progress:
        records += [p for p in root.glob('*/progress.json') if not (p.parent/'metrics.json').exists()]
    for metric in sorted(records):
        m=json.loads(metric.read_text())
        if not m.get('complete') and not include_progress:continue
        names=list(m['layers_by_stage_zero_based'])
        if any(not (metric.parent/f'{name}.mp4').exists() for name in names):continue
        indices=[round(i*(m['frames']-1)/4) for i in range(5)]
        thumbnail_width=256
        thumb_h=round(thumbnail_width*m['shape'][-2]/m['shape'][-1])
        sheet=Image.new('RGB',(5*thumbnail_width,len(names)*(thumb_h+24)),'white')
        draw=ImageDraw.Draw(sheet)
        for row,name in enumerate(names):
            source=metric.parent/f'{name}.mp4'
            iterator=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24')
            meta=next(iterator);w,h=meta['size'];seen=0;found=0
            if (w,h)!=(m['shape'][-1]*8,m['shape'][-2]*8):raise AssertionError(source)
            for i,data in enumerate(iterator):
                seen+=1
                if i not in indices:continue
                col=indices.index(i);found+=1;y=row*(thumb_h+24)
                frame=Image.frombytes('RGB',(w,h),data).resize((thumbnail_width,thumb_h),Image.Resampling.LANCZOS)
                sheet.paste(frame,(col*thumbnail_width,y+24))
                draw.text((col*thumbnail_width+3,y+5),f'{name} | f{i}',fill='black')
            if seen!=m['frames'] or found!=5:raise AssertionError(f'Frame mismatch:{source}')
        sheet.save(metric.parent/'five_frame_comparison.jpg',quality=92)
        print(metric.parent,flush=True)
    members={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name!='bundle_manifest.json' and path.suffix not in ('.lock','.tar','.gz'):
            members[path.relative_to(root).as_posix()]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'bundle_manifest.json').write_text(json.dumps(members,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--include-progress',action='store_true')
    a=p.parse_args();main(a.root,a.include_progress)
