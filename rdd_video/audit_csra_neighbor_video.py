"""Validate the completed video gate using all paired group-level measurements."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(path):
    data = json.loads(path.read_text())
    assert data['complete'] and not data['smoke']
    assert data['version'] == 'csra_video_front1_neighbor_feasibility_v2'
    assert data['front_keep'] == 1 and data['neighbor_rank'] == 128
    assert data['NFE'] == 3 and data['frames'] == 29 and data['vsa_sparsity'] == .8
    assert data['two_weight_sets_resident_for_all_paired_runs'] and not data['offload']
    for key in ('duplicate_model_max_abs','baseline_wrapper_max_abs','zero_init_neighbor_final_latent_max_abs'):
        assert data[key] == 0
    expected = {(case,variant) for case in ('832x448_spatial421','832x448_st421',
        '1280x768_spatial421','1280x768_st421') for variant in ('reader_only','reader_neighbor')}
    assert {(e['case'],e['variant']) for e in data['entries']} == expected and len(data['entries']) == 8
    rows = []; total = 0; peak = 0
    for entry in data['entries']:
        assert entry['front_keep'] == 1 and entry['prefix_depth'] == 6 and len(entry['groups']) == 12
        for j, group in enumerate(entry['groups']):
            assert group['order'] == ([0,6,6,0] if j%2 == 0 else [6,0,0,6])
            assert len(group['rows']) == 4
            for i,r in enumerate(group['rows']):
                depth = group['order'][i]
                assert r['depth'] == depth and r['front_keep'] == 1
                assert r['audit']['executed'] == [30,30,30-depth]
                assert r['audit']['skipped'] == ([] if depth == 0 else list(range(1,7)))
                assert r['audit']['bridge_calls'] == int(depth>0)
                assert r['neighbor_executed'] == bool(depth and entry['variant']=='reader_neighbor')
                assert r['wall_seconds'] > 0 and r['peak_allocated_gib'] < 24
                if depth:
                    assert r['audit']['bridge_seconds'] > 0 and r['audit']['capture_seconds'] > 0
                peak = max(peak,r['peak_allocated_gib']); total += 1
        rows.append(dict(case=entry['case'],variant=entry['variant'],baseline=entry['baseline_wall']['median'],
            candidate=entry['candidate_wall']['median'],latency_ci95=entry['paired_latency_ci95_percent'],
            bridge=entry['bridge_seconds']['median']))
    assert total == 384
    gated = [r for r in rows if r['case'].startswith('1280') and r['variant']=='reader_neighbor']
    passed = len(gated)==2 and all(r['latency_ci95']['lower'] >= 10 for r in gated)
    return dict(complete=True,timing_gate_passed=passed,measured_trajectories=total,peak_allocated_gib=peak,
        metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,
        warning='Untrained video feasibility on shared GPU, excludes VAE. Does not establish trained video quality or neuron-level causality. Reader-on/off were separate paired-to-baseline groups, not directly paired with each other.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='eval_results/csra_neighbor_video_feasibility_20260914/full')
    a=p.parse_args();root=Path(a.root)
    result=audit(root/'metrics.json')
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['timing_gate_passed']:raise SystemExit('New training gate failed')
