"""CPU verification of real cached clips, committed generation chunks and media."""
import copy
import hashlib
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[3]
G = WORK / 'gpu_checks_20261006'
sys.path.insert(0, str(WORK))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    import cv2
    from merge_video_generations import merge
    from summarize_paper_results import quality
    source = WORK / 'assets_20261006/small_mixkit/manifest14.jsonl'
    manifest = {r['id']: r for r in map(json.loads, source.read_text().splitlines())}
    video_hashes = {key: digest(Path(row['video'])) for key, row in manifest.items()}
    assert len(set(video_hashes.values())) == 14
    families = {}
    for family in ('wan', 'pf'):
        folder = WORK / 'fair_data_20261006' / ('fixture14_' + family)
        audit = json.loads((folder / 'cache_audit.json').read_text())
        assert audit['input_sha256'] == digest(source)
        assert audit['complete_manifest_processed'] and audit['counts'] == {'train': 12, 'validation': 2}
        identities = {}
        file_hashes = {}
        for clip in audit['committed_clips']:
            identity = clip['identity']
            original = manifest[identity['id']]
            assert identity == {**original, 'video_sha256': video_hashes[identity['id']]}
            identities[identity['id']] = identity
            for name, expected in clip['file_sha256'].items():
                assert digest(Path(name)) == expected, name
                file_hashes[name] = expected
        assert identities.keys() == manifest.keys()
        merged_entries: list[str] = []
        if family == 'wan':
            for split in ('train', 'validation'):
                paths = list((folder / 'controlled_wan' / split).glob('*.parquet'))
                assert len(paths) == audit['counts'][split]
                for path in paths:
                    assert digest(path) in file_hashes.values()
                merged_entries.extend(map(str, paths))
        else:
            for split in ('train', 'validation'):
                rows = list(map(json.loads, (folder / (split + '.jsonl')).read_text().splitlines()))
                assert len(rows) == audit['counts'][split]
                for row in rows:
                    assert row['split'] == split and row['text'] == manifest[row['id']]['caption']
                    assert row['latent'] in file_hashes and row['text_fea'] in file_hashes
                merged_entries.extend(row['id'] for row in rows)
        families[family] = dict(status='passed',
                                counts=audit['counts'],
                                hashed_cache_files=len(file_hashes),
                                merged_entries=len(merged_entries),
                                original_precision=audit['contracts'][0]['vae_weight_precision'])
    (G / 'REAL_DATA_FINAL_HASH_AUDIT.json').write_text(
        json.dumps(dict(status='passed',
                        synthetic=False,
                        unique_source_videos=14,
                        train_validation_video_hashes_disjoint=True,
                        families=families,
                        gpu_used=False),
                   indent=2))

    info = json.loads((WORK / 'paper_sources_20261006/VBench_full_info.json').read_text())
    prompts = list(dict.fromkeys(row['prompt_en'] for row in info))
    jobs = json.loads((G / 'quality_chunk_public_jobs.json').read_text())
    chunks = []
    for job in jobs:
        command = job['command']
        folder = Path(command[command.index('--output') + 1])
        value = json.loads((folder / 'metrics.json').read_text())
        offset = int(command[command.index('--prompt-start') + 1])
        assert value['latency_benchmark'] is False and value['warmup'] == 0 and value['repeats'] == 1
        assert value['prompt_start'] == offset and len(value['samples']) == 1
        row = value['samples'][0]
        assert row['caption'] == prompts[offset]
        assert row['name'] == f'prompt_{offset:04d}_seed_42'
        assert len(row['timings']) == 1 and row['frames'] == 81
        assert row.get('actual_nfe', row.get('velocity_evaluations')) == 50
        assert row['branch_forward_calls'] == 100
        chunks.append(
            dict(method=value['method'], output=str(folder), checkpoint=value['checkpoint'], prompt_start=offset))
    negative = {}
    trial = G / 'chunk_merge_negative_checks'
    trial.mkdir(exist_ok=True)
    first = G / 'quality_chunk_native_wan_0'
    second = G / 'quality_chunk_native_wan_1'
    try:
        merge([first, first], trial / 'duplicate_output')
    except ValueError:
        negative['duplicate_pair_rejected_before_writes'] = not (trial / 'duplicate_output').exists()
    bad = trial / 'bad_protocol'
    bad.mkdir(exist_ok=True)
    (bad / 'complete.json').write_text((second / 'complete.json').read_text())
    metrics = json.loads((second / 'metrics.json').read_text())
    metrics['cfg'] = 6.
    (bad / 'metrics.json').write_text(json.dumps(metrics))
    try:
        merge([first, bad], trial / 'mismatched_output')
    except ValueError:
        negative['different_cfg_rejected_before_writes'] = not (trial / 'mismatched_output').exists()
    merged = G / 'quality_chunk_native_wan_merged'
    assert len(json.loads((merged / 'metrics.json').read_text())['samples']) == 2
    statistics = json.loads((G / 'quality_chunk_ci_self_check.json').read_text())
    assert statistics['comparisons']['baseline']['paired_total_speedup'] is None
    assert statistics['comparisons']['baseline']['quality_complete'] is False
    score_file = G / 'cli_vbench_all16_current_v2/dimension_results.json'
    parsed = quality(score_file)
    original_scores = json.loads(score_file.read_text())['scores']
    assert len(parsed) == 16
    assert all(row['official_aggregate'] == original_scores[key][0] for key, row in parsed.items())
    undefined = trial / 'undefined_scores.json'
    value = copy.deepcopy(json.loads(score_file.read_text()))
    value['undefined_or_failed'] = ['color']
    undefined.write_text(json.dumps(value))
    try:
        quality(undefined)
    except ValueError:
        negative['undefined_official_metric_rejected'] = True
    assert len(negative) == 3 and all(negative.values())
    (G / 'QUALITY_CHUNK_CURRENT_ENTRYPOINTS.json').write_text(
        json.dumps(dict(status='passed',
                        chunks=chunks,
                        merged_real_samples=2,
                        one_trajectory_per_prompt_seed=True,
                        quality_run_not_used_as_latency=True,
                        official_aggregates_preserved=16,
                        negative_checks=negative,
                        quality_benchmark=False),
                   indent=2))

    files = set(map(Path, json.loads((G / 'current_media_files.json').read_text())))
    for folder in [Path(row['output'])
                   for row in chunks] + [G / 'vbench_native_full_resolution_v2', G / 'cli_native_rdd_current_sample']:
        files.update(folder.glob('*.mp4'))
    media = []
    for path in sorted(files):
        metadata = json.loads((path.parent / 'metrics.json').read_text())
        expected = next(row for row in metadata['samples'] if row['name'] == path.stem)
        stream = cv2.VideoCapture(str(path))
        fps = stream.get(cv2.CAP_PROP_FPS)
        frames = 0
        resolution = None
        while True:
            ok, frame = stream.read()
            if not ok:
                break
            resolution = list(frame.shape[:2])
            frames += 1
        stream.release()
        expected_frames = expected.get('frames', (expected['shape'][2] - 1) * 4 + 1)
        expected_resolution = expected.get('resolution', [expected['shape'][-2] * 8, expected['shape'][-1] * 8])
        assert frames == expected_frames and resolution == expected_resolution, str(path)
        assert abs(fps - metadata['fps']) < .01
        media.append(dict(file=str(path), frames=frames, resolution=resolution, fps=fps, sha256=digest(path)))
    (G / 'FINAL_MEDIA_DECODE_AUDIT.json').write_text(
        json.dumps(dict(status='passed', files=media, gpu_used=False), indent=2))
    print(
        json.dumps(dict(status='passed', real_videos=14, decoded_output_videos=len(media), quality_chunks=len(chunks))))


if __name__ == '__main__':
    main()
