"""Scientific, fixed frame-index grids; show all exported methods, no retouch."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg


def main(root):
    for path in sorted(Path(root).glob('video_*/metrics.json')):
        m=json.loads(path.read_text())
        if not m.get('complete'):continue
        names=list(m['layers_by_stage_zero_based'])
        if (path.parent/'baseline_fp32_decode.mp4').exists(): names.insert(0,'baseline_fp32_decode')
        indices=[round(i*(m['frames']-1)/4) for i in range(5)];rows=[]
        for name in names:
            source=path.parent/f'{name}.mp4';it=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24')
            meta=next(it);w,h=meta['size'];images=[];seen=0
            for i,data in enumerate(it):
                seen+=1
                if i in indices:
                    image=Image.frombytes('RGB',(w,h),data)
                    images.append(image.resize((320,round(h*320/w)),Image.Resampling.LANCZOS))
            if seen!=m['frames'] or len(images)!=5:raise AssertionError(f'Frame mismatch:{source}:{seen}')
            if (w,h)!=(m['shape'][-1]*8,m['shape'][-2]*8):raise AssertionError('Resolution mismatch')
            rows.append(images)
        height=rows[0][0].height;canvas=Image.new('RGB',(1600,len(rows)*(height+28)),'white')
        draw=ImageDraw.Draw(canvas)
        for r,(name,images) in enumerate(zip(names,rows)):
            for c,image in enumerate(images):
                y=r*(height+28);draw.text((c*320+4,y+5),f'{name}: frame{indices[c]}',fill='black')
                canvas.paste(image,(c*320,y+28))
        canvas.save(path.parent/'five_frame_comparison.jpg',quality=93)
        print(path.parent/'five_frame_comparison.jpg',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
