#!/usr/bin/env python3
"""Direct source reader/label parity, independent of generic resume fixtures."""
import argparse,hashlib,json,os,random,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from umdiff import native
from umdiff.cli import conditions
from umdiff.input_transforms import LocalMomentDataset
from umdiff.data_pipeline import NativeRelayDataset
from validate_fresh_process_lifecycle import manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    env=native.extract('dimr/datasets.py',['FeatureDataset','CFGDataset'],dict(Dataset=torch.utils.data.Dataset,os=os,np=np,random=random))
    rows=[]
    with tempfile.TemporaryDirectory(prefix='author_reader_contract_') as tmp:
        tmp=Path(tmp);moments=tmp/'moments';moments.mkdir();gen=np.random.default_rng(91)
        for i in range(11):
            value=np.empty(2,dtype=object);value[:]=[gen.normal(size=(8,4,4)).astype(np.float32),i*91];np.save(moments/f'{i}.npy',value)
        for probability in (0.,.1,1.):
            raw=env['FeatureDataset'](str(moments));original=env['CFGDataset'](raw,probability,1000) if probability else raw
            port=LocalMomentDataset(moments,cfg_dropout=probability);seen=[]
            for i in range(44):
                random.seed(981+i);x,y=original[i%11];after=random.getstate()
                random.seed(981+i);value=port[i%11]
                assert random.getstate()==after
                assert manifest(torch.as_tensor(x))==manifest(value['images']) and int(y)==int(value['labels'])
                # Reader CFG and main-loop CFG are alternatives, never both.
                kw,_=conditions({'conditioning':{'y':dict(kind='labels',classes=1000,allow_null=True,dropout=0.)}},
                    {'labels':value['labels'][None]},'cpu',1)
                assert int(kw['y'][0])==int(y);seen.append(int(y))
            rows.append(dict(method='dimr',p=probability,records=44,status='passed',reader_and_labels_byte_exact=True,python_rng_exact=True,null_labels=seen.count(1000),
                source_length=len(raw),fixture_length=len(port),length_deviation='Only records0..10 tested; source full dataset length retained but not iterated/downloaded.'))
        images=tmp/'relay_images';images.mkdir();labels=[]
        for i in range(5):
            name=f'{i}.png';Image.fromarray(gen.integers(0,256,(16,16,3),dtype=np.uint8)).save(images/name);labels.append([name,(i*7)%3])
        (images/'dataset.json').write_text(json.dumps({'labels':labels}));archive=tmp/'relay.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for f in images.iterdir():z.write(f,f.name)
        for xflip in (False,True):
            port=NativeRelayDataset(archive,resolution=16,use_labels=True,xflip=xflip)
            # Reader reuse is intentional. This separately verifies the public
            # wrapper's pixel/metadata conversion, not an independent ZIP parser.
            import importlib
            original=importlib.import_module('training.dataset').ImageFolderDataset(path=str(archive),resolution=16,use_labels=True,xflip=xflip)
            for i in range(len(original)):
                pixels,label=original[i];value=port[i]
                assert manifest(value)==manifest(dict(images=torch.from_numpy(pixels).float()/127.5-1,class_labels=torch.from_numpy(label)))
            rows.append(dict(method='relay',xflip=xflip,records=len(original),status='passed',pixels_and_source_dataset_json_labels_exact=True,source_reader_reused=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='passed',cases=rows,source_executions=native.AUDIT,gpu_used=False,temporary_payloads_deleted=True,
        dimr_reader_cfg_recipe=dict(data=dict(kind='moments_npy',cfg_dropout=.1,null_label=1000),conditioning=dict(y=dict(kind='labels',classes=1000,allow_null=True,dropout=0.))),
        scope='Direct official reader getitem/CFG and wrapper input contracts; generated local files. Separate existing tests cover actual worker traversal/resume. No real dataset/pretrained codec claim.',harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',cases=len(rows))))
if __name__=='__main__':main()
