"""Regenerate small common-backbone examples; not paper best-result recipes."""
import json
from pathlib import Path
from umdiff.cross_modal import recipe,METHODS

ROOT=Path(__file__).resolve().parents[1]

def main():
    directory=ROOT/'configs/cross_modal';directory.mkdir(exist_ok=True)
    examples={}
    for method in METHODS:
        for video in (False,True):
            name=f'{method}_{"video" if video else "image"}_grid_dit'
            examples[name]=recipe(method,[3,*([5] if video else []),32,32])
    for policy in ('speed','bottleneck'):
        for video in (False,True):
            examples[f'flow_{policy}_{"video" if video else "image"}']=recipe('flow',[3,*([5] if video else []),32,32],inference_policy=policy)
    for backbone in ('grid_dit','wan_grid'):
        examples[f'pyramidal_wan_spacetime_{backbone}']=recipe('pyramidal_wan_path',[4,9,32,32],backbone,
                    grids=[[3,8,8],[5,16,16],[9,32,32]],first_frame=True)
    examples['rdd_video_dit_grid']=recipe('rdd',[4,5,32,32],'dit_grid')
    examples['rdd_video_decoupled']=recipe('rdd',[4,5,32,32],decoupled=True)
    examples['rdd_video_separate_stages']=recipe('rdd',[4,5,32,32],separate_stages=True)
    for name,config in examples.items():
        (directory/(name+'.json')).write_text(json.dumps(config,indent=2)+'\n')
    (directory/'manifest.json').write_text(json.dumps({'purpose':'small executable controlled recipes, not official best settings',
          'configs':sorted(name+'.json' for name in examples),'device_validation':'CPU','no_asset_downloads':True},indent=2)+'\n')
    print(json.dumps({'directory':str(directory),'count':len(examples)}))

if __name__=='__main__':main()
