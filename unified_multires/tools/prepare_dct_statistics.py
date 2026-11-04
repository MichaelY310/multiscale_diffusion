#!/usr/bin/env python3
"""Native percentile bound / entropy fields from three LOCAL coefficient arrays."""
import argparse,sys,json,io,contextlib,re,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import numpy as np
from umdiff import native
def prepare(paths,block,tau=98.25,eta=None):
    function=native.extract('dctdiff/DCT_datasets_statis.py',['DCT_statis_from_array'],dict(np=np))['DCT_statis_from_array'];raw=[]
    def call(path,bound=None):
        array=np.load(path,mmap_mode='r')
        if array.ndim!=2 or array.shape[1]!=block*block:raise ValueError('Each array must contain [number of blocks, block**2] raw raster-order DCT coefficients')
        text=io.StringIO()
        with contextlib.redirect_stdout(text):function(array_path=str(path),block_sz=block,tau=tau,eta=bound)
        raw.append(text.getvalue());return text.getvalue()
    if eta is None:eta=float(re.search(r'eta is ([^\n]+)',call(paths[0])).group(1))
    result={'block':block,'frequencies':block*block,'Y_bound':[eta],'statistic':'author entropy histogram fields named *_std, not ordinary standard deviation','raw_native_output':raw}
    for key,path in zip(('Y_std','Cb_std','Cr_std'),paths):
        text=call(path,eta);values=re.search(r'entropy: (.+)',text).group(1);values=re.sub(r'np\.float64\(([^)]*)\)',r'\1',values);result[key]=ast.literal_eval(values)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--arrays',nargs=3,required=True,metavar=('Y','CB','CR'));p.add_argument('--block',type=int,required=True);p.add_argument('--tau',type=float,default=98.25);p.add_argument('--eta',type=float);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=prepare(a.arrays,a.block,a.tau,a.eta);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(str(a.output))
if __name__=='__main__':main()
