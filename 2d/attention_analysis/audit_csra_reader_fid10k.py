"""Read-only experiment audit plus fixed, unselected comparison artifacts.

Never changes sampling/training, checkpoints, NPZ or sample pixels. CPU only.
Use --sampling-only while the independent ADM evaluation is still running.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from sample_csra_reader_fid10k import ROOT,ARMS,BASE_SHA,READER,READER_SHA,protocol,batches,check_meta,sha,write_json


def main(args):
    root=args.root
    assert sha('checkpoints/RDD-Unified-L/0400000.pt')==BASE_SHA and sha(READER)==READER_SHA
    done=json.loads((root/'sampling_complete.json').read_text())
    assert done['complete'] and done['protocol']==protocol(False)
    provenance={k:v for k,v in done.items() if k not in ('complete','warning')}
    for key,path in [('sampler_sha256','sample_csra_reader_fid10k.py'),
                     ('shared_sampler_sha256','sample_csra_adapter_preview.py'),
                     ('labelled_wrapper_sha256','sample_csra_prefix_quality_pilot.py')]:
        assert provenance[key]==sha(Path(__file__).parent/path)
    counts={a:0 for a in ARMS};chunks=0;preview_hashes={};reader_configs=[]
    for rank in range(4):
        shard=root/f'shard_{rank}'
        assert json.loads((shard/'complete.json').read_text())==dict(complete=True,**provenance)
        reader_configs.append(json.loads((shard/'reader_config.json').read_text()))
        # Fixed first8 from each shard, in generation order, not chosen by quality.
        grid=Image.new('RGB',(530,30+8*282),'white');draw=ImageDraw.Draw(grid)
        draw.text((5,8),'Original RDD',fill='black');draw.text((270,8),'Reader (skip blocks1-6)',fill='black')
        for arm in ARMS:
            column=ARMS.index(arm)
            for i,n,seed in batches(rank):
                path=shard/arm/f'chunk_{i:03d}.npz'
                meta=json.loads(path.with_suffix('.json').read_text())
                check_meta(meta,path,i,n,seed,provenance)
                ref=json.loads((shard/ARMS[0]/path.with_suffix('.json').name).read_text())
                assert meta['labels']==ref['labels'] and meta['endpoints']==ref['endpoints']
                assert set(meta['endpoints'])=={'8','16'}
                with np.load(path,allow_pickle=False) as data:
                    pixels=data['arr_0'];labels=data['arr_1']
                    assert pixels.shape==(n,256,256,3) and pixels.dtype==np.uint8
                    assert labels.tolist()==meta['labels'] and all(0<=x<1000 for x in labels)
                    if i==0:
                        for j in range(8):
                            preview=shard/arm/f'preview_{j:02d}.png'
                            with Image.open(preview) as im:
                                assert np.array_equal(np.asarray(im.convert('RGB')),pixels[j])
                                grid.paste(im.convert('RGB'),(column*265,30+j*282))
                            draw.text((column*265+3,288+j*282),f'shard{rank} sample{j} class{int(labels[j])}',fill='black')
                            preview_hashes[str(preview)]=sha(preview)
                counts[arm]+=n;chunks+=1
        grid.save(shard/'comparison_grid.png')
    assert chunks==320 and counts=={a:10000 for a in ARMS}
    assert all(c==reader_configs[0] for c in reader_configs)
    assert reader_configs[0]['checkpoint_sha256']==READER_SHA
    assert all(sha(p)==s for p,s in reader_configs[0]['code_sha256'].items())
    for arm in ARMS:
        assembled=json.loads((root/f'{arm}_assembled.json').read_text())
        assert assembled['complete'] and assembled['count']==10000 and assembled['provenance']==provenance
        assert assembled['sha256']==sha(root/f'{arm}.npz')
    metric_provenance=json.loads((root/'metric_provenance.json').read_text())
    assert all(sha(p)==s for p,s in metric_provenance.items())
    report=dict(complete=True,kind='sampling_audit',chunks=chunks,counts=counts,
        model_sha256=dict(original_rdd=BASE_SHA,reader=READER_SHA),provenance=provenance,
        metric_provenance=metric_provenance,preview_sha256=preview_hashes,
        checked='All320chunk SHA, pixel shape/dtype, labels/range/count, paired seeds and original8/16 endpoint hashes; fixed64previews match chunk pixels; finalNPZ SHA; source/checkpoint/metric SHA.',
        warning='Sampling completion alone is not quality equivalence, a speed result, or final50K FID.')
    write_json(root/'sampling_audit.json',report)
    if not args.sampling_only:
        final=json.loads((root/'complete.json').read_text())
        metrics_path=root/'metrics/summary_fid10k.json'
        assert final['complete'] and final['summary_sha256']==sha(metrics_path)
        metrics=json.loads(metrics_path.read_text())['datasets']
        names={'Inception Score','FID','sFID','Precision','Recall'}
        assert set(metrics)==set(ARMS)
        assert all(set(v)==names and all(math.isfinite(x) for x in v.values()) for v in metrics.values())
        delta={k:metrics['reader'][k]-metrics['original_rdd'][k] for k in sorted(names)}
        write_json(root/'quality_audit.json',dict(complete=True,sampling_audit_sha256=sha(root/'sampling_audit.json'),
            metrics=metrics,reader_minus_original=delta,summary_sha256=sha(metrics_path),
            warning='Independent paired same-N10K point estimates; no confidence interval or equivalence claim. Not comparable to published50K absolute FID. Video quality remains untested.'))
    print(json.dumps(dict(audit_complete=True,mode='sampling_only' if args.sampling_only else 'final',chunks=chunks,counts=counts)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--sampling-only',action='store_true')
    main(parser.parse_args())
