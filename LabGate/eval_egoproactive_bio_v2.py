#!/usr/bin/env python3
"""Bio v2: structured reasons, cleaned chat history, regular current video, causal state."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from egoproactive_bio import summarize
from eval_egoproactive_bio import write_json
from proactive_v2 import historical_indices, regular_window, run_step, update_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run', type=Path, required=True, help='Use recorded dataset and model paths from v1')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--history-modes', nargs='+', choices=['gold', 'rollout'], default=['gold', 'rollout'])
    p.add_argument('--target-fps', type=float, default=2.0)
    p.add_argument('--max-current-frames', type=int, default=32)
    p.add_argument('--history-images', type=int, default=4)
    p.add_argument('--max-pixels', type=int, default=256 * 28 * 28)
    args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error('Output directory must be empty')
    if args.history_images < 0:
        p.error('history-images must be nonnegative')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from proactive_models import ProactiveQwenVL

    baseline = json.loads((args.baseline_run / 'run.json').read_text())
    dataset = Path(baseline['args']['dataset'])
    annotation = dataset / 'egoproactive/egoproactive_bio_val.jsonl'
    sha = hashlib.sha256(annotation.read_bytes()).hexdigest()
    if sha != baseline['annotation_sha256']:
        raise ValueError('Dataset changed since baseline; record a fresh baseline first')
    sample = json.loads(annotation.read_text())
    decisions = json.loads((dataset / 'metadata/composition.json').read_text())['decisions']
    assert len(decisions) == len(sample['answers']) == len(sample['video_intervals']) == len(sample['dialog'])
    assert all(d['target_interval_sec'] == interval and d['answer'] == answer
               for d, interval, answer in zip(decisions, sample['video_intervals'], sample['answers']))
    count = args.limit or len(decisions)
    if not 0 < count <= len(decisions):
        p.error('Invalid limit')
    np.random.seed(0)
    torch.manual_seed(0)
    metadata = {
        'status': 'preparing', 'version': 'bio_v2', 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'models': {k: baseline['args'][k] for k in ['judger', 'expert']},
        'dataset': str(dataset), 'annotation_sha256': sha, 'host': platform.node(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'packages': {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'decord', 'numpy']},
        'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                         for name in ['eval_egoproactive_bio_v2.py', 'proactive_v2.py', 'proactive_models.py',
                                      'prompts.py', 'models.py', 'egoproactive_bio.py']},
        'state_policy': 'last two small-model observations, updated even when silent; no gold step labels',
        'history_policy': 'actual chat roles, control tags removed, at most four turns after initial query',
        'timing': 'wall includes actual shared preparation cost + processor/generation; loading separate; no warmup',
    }
    write_json(args.out_dir / 'run.json', metadata)
    t0 = time.perf_counter()
    reader = VideoReader(baseline['video'], ctx=cpu(0), num_threads=2)
    source_fps = float(reader.get_avg_fps())
    prepared = []
    for i, (start, end) in enumerate(sample['video_intervals'][:count]):
        prep = time.perf_counter()
        indices, fps = regular_window(start, end, source_fps, len(reader), args.target_fps, args.max_current_frames)
        historical = historical_indices(start, source_fps, args.history_images)
        frames = reader.get_batch(indices).asnumpy()
        past = reader.get_batch(historical).asnumpy() if historical else []
        visual = {'frames': frames, 'history_frames': past, 'fps': fps,
                  'timestamps_sec': [x / source_fps for x in indices],
                  'history_timestamps_sec': [x / source_fps for x in historical],
                  'interval_sec': [start, end]}
        assert all(t < end for t in visual['timestamps_sec'])
        assert all(t < start for t in visual['history_timestamps_sec'])
        info = {k: v for k, v in visual.items() if k not in ['frames', 'history_frames']}
        info.update(decision_id=decisions[i]['decision_id'], frame_indices=indices, historical_indices=historical,
                    n_frames_in=len(indices) + len(historical), n_frames_kept=len(indices) + len(historical),
                    current_frames=len(indices), history_images=len(historical), latency_prepare_s=time.perf_counter() - prep)
        prepared.append((visual, info))
    del reader
    metadata['preparation_wall_s'] = time.perf_counter() - t0
    write_json(args.out_dir / 'inputs.json', [info for _, info in prepared])
    t0 = time.perf_counter()
    judger = ProactiveQwenVL(metadata['models']['judger'], max_pixels=args.max_pixels, max_new_tokens=160)
    expert = ProactiveQwenVL(metadata['models']['expert'], max_pixels=args.max_pixels, max_new_tokens=128)
    metadata['model_load_wall_s'] = time.perf_counter() - t0
    metadata['device_maps'] = {k: {key: str(v) for key, v in model.model.hf_device_map.items()}
                               for k, model in [('judger', judger), ('expert', expert)]}
    metadata['status'] = 'running'
    write_json(args.out_dir / 'run.json', metadata)
    summaries = {}
    for mode in dict.fromkeys(args.history_modes):
        history = [{'role': 'user', 'text': sample['query']}]
        state, rows = [], []
        with (args.out_dir / f'{mode}.predictions.jsonl').open('w') as output:
            for i, (visual, info) in enumerate(prepared):
                prior = sample['dialog'][i] if mode == 'gold' else list(history)
                t0 = time.perf_counter()
                result = run_step(judger, expert, visual, sample['query'], prior, state)
                wall = time.perf_counter() - t0 + info['latency_prepare_s']
                gold = sample['answers'][i]
                row = {**info, **result, 'latency_wall_s': wall, 'history_mode': mode,
                       'gt_label': 'interrupt' if gold.startswith('$interrupt$') else 'silent',
                       'gt_answer': gold, 'phase': decisions[i]['phase'], 'pair_id': decisions[i]['pair_id'],
                       'task_for_audit_only': sample['task'][i]}
                update_state(state, result, info['interval_sec'][1])
                if result['fired']:
                    history.append({'role': 'assistant', 'text': result['answer']})
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                score = {**summarize(rows), 'schema_valid_count': sum(r['trigger']['schema_valid'] for r in rows),
                         'reason_counts': {reason: sum(r['fired'] and r['trigger']['reason_type'] == reason for r in rows)
                                           for reason in ['safety_warning', 'action_error', 'next_step', 'user_query', 'unknown', 'none']}}
                write_json(args.out_dir / f'{mode}.summary.json', {'complete': len(rows) == count, **score})
                print(f"{mode} {i + 1}/{count} gt={row['gt_label']} pred={row['pred_label']} "
                      f"reason={result['trigger']['reason_type']} observation={result['trigger']['observation']} "
                      f"guidance={result['guidance']}", flush=True)
        summaries[mode] = score
    metadata['status'] = 'complete'
    metadata['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(args.out_dir / 'run.json', metadata)
    write_json(args.out_dir / 'summary.json', summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
