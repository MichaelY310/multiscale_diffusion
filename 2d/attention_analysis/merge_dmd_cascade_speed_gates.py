"""Assemble the two selected, independently baseline-paired spatial gates."""
import argparse
import json
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    result={'complete':True,'baseline':'same Spatial421 RDD + native VSA20, no reuse; identical BF16 VAE',
            'criterion':'paired sample+VAE mean AND bootstrap lower bound >=20%',
            'shared_gpu':True,'candidates':{}}
    for family,folder in [('high_heavy','video_spatial_speed'),('balanced_plus','video_spatial_plus_speed')]:
        source=root/folder/'speed_gate.json'
        if not source.exists():result['complete']=False;continue
        gate=json.loads(source.read_text());entry=gate['candidates'][family]
        result['candidates'][family]={**entry,'source_gate':str(source)}
        result['complete'] &= gate['complete'] and entry['passed']
    (root/'speed_gate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
