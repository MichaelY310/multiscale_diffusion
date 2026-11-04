"""Predeclared six-block layouts; intervals are zero-based [start,end)."""
LAYOUTS = {
    'front0_6': ((0, 6),),
    'front3_6': ((3, 9),),
    'front6_6': ((6, 12),),
    'split3_3': ((0, 3), (6, 9)),
}


def layout(name):
    spans = LAYOUTS[name]
    skipped = [i for start,end in spans for i in range(start,end)]
    if len(skipped) != 6 or len(set(skipped)) != 6 or any(i not in range(24) for i in skipped):
        raise ValueError('All layouts must skip exactly six of24 blocks')
    return dict(name=name, spans=[list(s) for s in spans], skipped_layers=skipped,
        executed_layers=[i for i in range(24) if i not in skipped], source_layer=13,
        readers=len(spans), skip_count=6)
