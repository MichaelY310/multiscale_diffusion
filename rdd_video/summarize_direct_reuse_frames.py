"""Scientific contact sheets: same5 frame indices, every method, no retouching."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg


def main(root):
    for case in ('video_832','video_1280','video_temporal221'):
        folder=Path(root)/case;names=('baseline','reuse6','reuse8','reuse10');indices=(0,7,14,21,28)
        rows=[]
        for name in names:
            it=imageio_ffmpeg.read_frames(str(folder/f'{name}.mp4'),pix_fmt='rgb24');meta=next(it)
            w,h=meta['size'];frames=[]
            for i,data in enumerate(it):
                if i in indices:
                    frames.append(Image.frombytes('RGB',(w,h),data).resize((320,round(320*h/w)),Image.Resampling.LANCZOS))
            if len(frames)!=5:raise ValueError('Expected29frames and5 fixed selections')
            rows.append(frames)
        height=rows[0][0].height;canvas=Image.new('RGB',(1600,4*(height+28)),'white');draw=ImageDraw.Draw(canvas)
        for r,(name,frames) in enumerate(zip(names,rows)):
            for c,frame in enumerate(frames):
                y=r*(height+28);draw.text((c*320+5,y+5),f'{name} / frame {indices[c]}',fill='black')
                canvas.paste(frame,(c*320,y+28))
        canvas.save(folder/'five_frame_comparison.jpg',quality=92)
        print(folder/'five_frame_comparison.jpg',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915');main(p.parse_args().root)
