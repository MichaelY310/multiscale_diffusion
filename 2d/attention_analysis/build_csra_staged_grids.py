"""Arrange unmodified fixed first8 images from both existing3-column grids."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('eval_results/csra_staged_front_review_20260914')


def build():
    columns = [('direct_control','original_dense','Original RDD'),
               ('direct_control','aligned','Direct1600 / source'),
               ('staged','aligned','Staged1600 / source'),
               ('direct_control','zero','Direct1600 / zero'),
               ('staged','zero','Staged1600 / zero')]
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    except OSError: font = ImageFont.load_default()
    for rank in range(4):
        canvas = Image.new('RGB',(1280,2100),'white')
        draw = ImageDraw.Draw(canvas)
        for col,(mode,arm,label) in enumerate(columns):
            draw.text((col*256+5,16),label,font=font,fill='black')
            for i in range(8):
                path = Path(f'eval_results/csra_staged_front_{mode}_quality256_20260914')/f'shard_{rank}'/arm/f'{i:03d}.png'
                with Image.open(path) as picture:
                    if picture.size != (256,256): raise ValueError('No rescaling allowed')
                    canvas.paste(picture.convert('RGB'),(col*256,52+i*256))
        canvas.save(OUT/f'comparison_shard{rank}.png')


if __name__ == '__main__': build()
