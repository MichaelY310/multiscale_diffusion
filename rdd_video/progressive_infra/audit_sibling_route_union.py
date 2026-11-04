"""CPU-only exact native-route sibling overlap; no speed/HBM claims."""
import argparse
import json
from pathlib import Path
import numpy as np


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), min=float(x.min()), max=float(x.max()))


def main(a):
    archive = np.load(a.archive)
    hw, heads = 240, 12
    cases = {(1, 2): [], (2, 2): [], (2, 4): []}
    root_keeps = []
    for layer in range(30):
        root = np.unpackbits(archive[f's0_l{layer}'], count=heads*hw*hw).reshape(heads, hw, hw).astype(bool)
        root_keeps.append(root.sum(-1, dtype=np.int32).ravel())
        for stage in (1, 2):
            ratio = 2**stage
            n = hw*ratio
            m = np.unpackbits(archive[f's{stage}_l{layer}'], count=heads*n*n).reshape(heads, n, n).astype(bool)
            siblings = m.reshape(heads, ratio, hw, n).transpose(0, 2, 1, 3)
            for group in (2, 4):
                if ratio % group:
                    continue
                grouped = siblings.reshape(heads, hw, ratio//group, group, n)
                union = grouped.any(axis=3).sum(-1, dtype=np.int32)
                selected = grouped.sum(axis=(3, 4), dtype=np.int32)
                cases[(stage, group)].append((union.ravel(), selected.ravel()))
    root_keep = np.concatenate(root_keeps)
    result = []
    for (stage, group), items in cases.items():
        union = np.concatenate([x[0] for x in items])
        selected = np.concatenate([x[1] for x in items])
        result.append(dict(stage=stage, group=group, groups=int(union.size),
            selected_tiles_per_row=describe(selected/group), union_tiles=describe(union),
            MMA_pair_inflation=describe(group*union/selected),
            MMA_fill=describe(selected/(group*union)),
            aggregate_MMA_inflation=float(group*union.sum()/selected.sum()),
            logical_KV_load_ratio_vs_separate_Q64=float(union.sum()/selected.sum()),
            lifted_C2F_union_tiles=describe(root_keep*2**stage),
            lifted_C2F_MMA_fill=1., lifted_C2F_logical_KV_load_ratio=1./group))
    report = dict(complete=True, source=str(a.archive), root_grid=[1, 12, 20],
        warning='Binary support audit only: union/per-row masks and construction cost require a separate real kernel benchmark. Not latency or physical HBM traffic.', rows=result)
    print(json.dumps(report, indent=2))
    if a.output:
        destination = Path(a.output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--output')
    main(p.parse_args())
