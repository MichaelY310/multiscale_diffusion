"""Strict fixed-budget 2x2 refinement; never silently migrate a resume file."""
PROTOCOL = 'front1_neighbor2x2_direct1600_resetadam_to2400_v1'
FORMAT = 'csra_front1_skip6_spatial_neighbor_v1'
PARENT_SHA = {
    'aligned': '3f8528337d11a9526c3a2379ca002d22bc943ec2458aaf3a658119acda8c70d8',
    'zero': '836176c130af9f8a6626d0e178485abac6cbf603498d6a33474f64dd250da640',
}
OPTIMIZER_POLICY = 'reset_all_AdamW_moments_at_parent1600_in_all_four_arms'
FIXED_ARGS = dict(seed=20279000, data_root='eval_results/csra_rollout_data4k_20260912',
                  hidden_root='eval_results/csra_hidden_sources_20260913',
                  base_ckpt='checkpoints/RDD-Unified-L/0400000.pt',
                  initializer='eval_results/csra_front_keep6_information_20260914/front1/selected_ridge.pt')


def check_parent(ck, arm, digest, base_sha, init_sha):
    if (digest != PARENT_SHA[arm] or ck.get('format') != 'csra_front1_skip6_conditional_v1'
            or ck.get('training_protocol') != 'front1_bridge_warmup800_joint800_v1'
            or ck.get('step') != 1600 or ck.get('front_keep') != 1
            or ck.get('prefix_depth') != 6 or ck.get('source_layer') != 13
            or ck.get('base_sha256') != base_sha or ck.get('initializer_sha256') != init_sha
            or ck['args'].get('coarse_source') != arm or ck['args'].get('mode') != 'direct_control'
            or any(ck['args'].get(k) != v for k, v in FIXED_ARGS.items())):
        raise ValueError('Require the immutable matching direct-control fixed1600 parent')


def check_new(ck, args, code_sha, base_sha, init_sha, require_final=False):
    if any(args.get(k) != v for k, v in FIXED_ARGS.items()) or args.get('neighbor') not in ('off','on'):
        raise ValueError('Changed fixed new-pilot arguments')
    if (ck.get('format') != FORMAT or ck.get('training_protocol') != PROTOCOL
            or ck.get('optimizer_policy') != OPTIMIZER_POLICY or ck.get('code_sha256') != code_sha
            or ck.get('base_sha256') != base_sha or ck.get('initializer_sha256') != init_sha
            or ck.get('front_keep') != 1 or ck.get('prefix_depth') != 6
            or ck.get('source_layer') != 13 or ck.get('neighbor_rank') != 128
            or ck.get('parent', {}).get('sha256') != PARENT_SHA[args['coarse_source']]
            or not 1600 <= ck.get('step', 0) <= 2400):
        raise ValueError('New checkpoint protocol/code/parent contract mismatch')
    for key in (*FIXED_ARGS, 'coarse_source', 'neighbor'):
        if ck['args'].get(key) != args.get(key):
            raise ValueError(f'New run/resume changed {key}')
    if require_final and (ck['step'] != 2400 or ck['args'].get('smoke')):
        raise ValueError('Evaluation requires fixed2400 formal checkpoint')


def source_paths():
    return ['2d/attention_analysis/train_csra_neighbor_pair.py',
            '2d/attention_analysis/csra_neighbor_protocol.py',
            '2d/attention_analysis/csra_neighbor_student.py',
            '2d/attention_analysis/csra_front_keep_student.py',
            '2d/attention_analysis/csra_prefix_student.py',
            'rdd_video/attention_analysis/csra_fine_neighbor.py']
