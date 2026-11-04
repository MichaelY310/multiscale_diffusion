"""Fixed-budget schedule, separate from model code and old checkpoint contracts."""
PROTOCOL = 'front1_bridge_warmup800_joint800_v1'


def phase(mode, completed_steps):
    if mode not in ('staged', 'direct_control') or not 0 <= completed_steps < 1600:
        raise ValueError('Unknown mode or step outside fixed1600 budget')
    return 'bridge_only' if mode == 'staged' and completed_steps < 800 else 'joint'


def check_parent(ck, arm, base_sha, initializer_sha):
    if (ck.get('format') != 'csra_front1_skip6_conditional_v1' or ck.get('step') != 800
            or ck.get('front_keep') != 1 or ck.get('prefix_depth') != 6 or ck.get('source_layer') != 13
            or ck.get('base_sha256') != base_sha or ck.get('initializer_sha256') != initializer_sha
            or ck['args']['coarse_source'] != arm or ck.get('training_protocol') is not None):
        raise ValueError('Direct control must continue the original matching front1 fixed800')
    expected = dict(seed=20279000, data_root='eval_results/csra_rollout_data4k_20260912',
                    hidden_root='eval_results/csra_hidden_sources_20260913')
    if any(ck['args'].get(k) != v for k, v in expected.items()):
        raise ValueError('Direct control data/seed mismatch')
