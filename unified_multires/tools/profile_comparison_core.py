"""Inspect XL tensor shapes and scalar-path arithmetic on meta tensors, no GPU."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
from umdiff import build

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'configs/comparison_imagenet256'
OUT = ROOT/'evidence/main_comparison_20261013'
torch.set_num_threads(2)

anchor = json.loads((CONFIGS/'main/rdd_ddpm_xl_legacy.json').read_text())
with torch.device('meta'):
    original = build('model', anchor['method']['model'])
    original_shapes = {k:list(v.shape) for k,v in original.state_dict().items()}
    costs = {}
    for r in (2,4,8,16,32):
        original.zero_grad(set_to_none=True)
        with FlopCounterMode(display=False) as counter:
            output = original(torch.randn(1,4,r,r), torch.ones(1), torch.zeros(1,dtype=torch.long))
            output.square().mean().backward()
        costs[str(r)] = counter.get_total_flops()
    del original, output
    rows = []
    for path in sorted(CONFIGS.glob('*/*.json')):
        spec = json.loads(path.read_text())
        model = build('model',spec['method']['model'])
        if spec['method']['model']['name']=='ssd_grid':
            # SSDGridModel's wrapped network is .model.
            core = model.model.core
        else:
            core = getattr(model,'core',model)
        shapes = {k:list(v.shape) for k,v in core.state_dict().items()}
        assert shapes == original_shapes, path
        rows.append(dict(id=spec['id'],suite=spec['suite'],
                         parameters=sum(p.numel() for p in model.parameters()),
                         core_parameters=sum(p.numel() for p in core.parameters()),
                         legacy_core_parameter_shapes_exact=True,
                         config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del model, core
        gc.collect()

OUT.mkdir(parents=True,exist_ok=True)
report = dict(status='passed',cuda_initialized=torch.cuda.is_initialized(),
              scope='Full XL meta construction; exact legacy core tensor shapes. No full-size numerical forward/backward claim.',
              configs=rows)
(OUT/'ARCHITECTURE.json').write_text(json.dumps(report,indent=2)+'\n')
flops = dict(status='passed',device='meta',cuda_initialized=torch.cuda.is_initialized(),
             torch_version=torch.__version__,batch=1,scalar_full_xl_forward_backward_flops=costs,
             multiply_add_counts_as=2,
             rdd_three_stage_to_fixed32_ratio=sum(costs[str(r)] for r in (8,16,32))/costs['32'],
             scope='PyTorch registered FLOP-counter operators for backbone forward and parameter backward; excludes VAE, data, optimizer/EMA and uncounted elementwise operators. Not latency, not total system FLOPs, and not a profile of joint-band adapters.')
(OUT/'SCALAR_CORE_FLOPS.json').write_text(json.dumps(flops,indent=2)+'\n')
print(json.dumps({'architecture_configs':len(rows),**flops}))
