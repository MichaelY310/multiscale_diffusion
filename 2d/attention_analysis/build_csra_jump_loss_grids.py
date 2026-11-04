"""Lay out existing unmodified scientific sample PNGs, fixed first8 per shard."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def build(root):
    columns = [('quality_old','original_dense','Original RDD'),
               ('quality_old','aligned','Old loss / source'),
               ('quality_jump','aligned','Jump loss / source'),
               ('quality_old','zero','Old loss / zero'),
               ('quality_jump','zero','Jump loss / zero')]
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    except OSError:
        font = ImageFont.load_default()
    for rank in range(4):
        canvas=Image.new('RGB',(256*len(columns),52+8*256),'white')
        draw=ImageDraw.Draw(canvas)
        for col,(loss,arm,label) in enumerate(columns):
            draw.text((col*256+8,16),label,font=font,fill='black')
            for i in range(8):
                path=root/loss/f'shard_{rank}'/arm/f'{i:03d}.png'
                with Image.open(path) as image:
                    if image.size!=(256,256):raise ValueError('No resizing of experimental images allowed')
                    canvas.paste(image.convert('RGB'),(col*256,52+i*256))
        out=root/f'factorial_comparison_shard{rank}.png'
        canvas.save(out)
        print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('eval_results/csra_prefix6_jump_loss_20260914'))
    build(p.parse_args().root)
