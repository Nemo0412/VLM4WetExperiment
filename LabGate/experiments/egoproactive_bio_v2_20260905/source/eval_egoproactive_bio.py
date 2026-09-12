#!/usr/bin/env python3
"""Evaluate the LabGate small→large cascade on EgoProactive-Bio decision points."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from egoproactive_bio import context_text, run_gated, sample_indices, summarize


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--judger', default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--expert', default='Qwen/Qwen2.5-VL-32B-Instruct')
    parser.add_argument('--history-modes', nargs='+', choices=['gold', 'rollout'], default=['gold', 'rollout'])
    parser.add_argument('--window-frames', type=int, default=8)
    parser.add_argument('--prefix-frames', type=int, default=8)
    parser.add_argument('--tau-reproj', type=float, default=0.12)
    parser.add_argument('--no-reprojection', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='Decode and validate inputs; no inference or accuracy claims')
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Output directory must be empty to avoid mixing runs')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from pipeline import reproject_keep
    from models import QwenVL

    np.random.seed(0)
    torch.manual_seed(0)
    annotation_path = args.dataset / 'egoproactive/egoproactive_bio_val.jsonl'
    samples = [json.loads(line) for line in annotation_path.read_text().splitlines() if line.strip()]
    if len(samples) != 1:
        raise ValueError('This evaluator currently expects the single-video Bio draft')
    sample = samples[0]
    decisions = json.loads((args.dataset / 'metadata/composition.json').read_text())['decisions']
    if not len(sample['answers']) == len(sample['video_intervals']) == len(sample['dialog']) == len(decisions):
        raise ValueError('Misaligned decisions')
    for d, interval, answer in zip(decisions, sample['video_intervals'], sample['answers']):
        if d['target_interval_sec'] != interval or d['answer'] != answer:
            raise ValueError('Composition and annotations disagree')
    video = args.dataset / 'egoproactive/val' / sample['video_path']
    vr = VideoReader(str(video), ctx=cpu(0), num_threads=2)
    fps = float(vr.get_avg_fps())
    count = args.limit or len(decisions)
    if not 0 < count <= len(decisions):
        raise ValueError('limit is out of range')
    metadata = {
        "status": "preparing", "started_at_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_version": (args.dataset / 'VERSION').read_text().strip(),
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "video": str(video), "video_fps": fps, "video_frames": len(vr),
        "source_commit": subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip(),
        "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), Path(__file__).with_name('egoproactive_bio.py'),
                                    Path(__file__).with_name('models.py'), Path(__file__).with_name('pipeline.py'),
                                    Path(__file__).parent.parent / 'FineBioWhen2See/gate.py']},
        "host": platform.node(), "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
        "packages": {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy', 'decord', 'accelerate']},
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "protocol": "static SOP; no current/future annotations supplied to models",
        "timing": "decision_wall includes decoding, reprojection, processor and generation; excludes model loading; no warmup",
    }
    write_json(args.out_dir / 'run.json', metadata)
    prepared = []
    preparation_start = time.perf_counter()
    for i, (start, end) in enumerate(sample['video_intervals'][:count]):
        t0 = time.perf_counter()
        indices = sample_indices(start, end, fps, len(vr), args.window_frames, args.prefix_frames)
        frames = vr.get_batch(indices).asnumpy()
        if args.no_reprojection:
            keep = list(range(len(frames)))
        else:
            _, keep = reproject_keep(frames, args.tau_reproj)
            # The decision must see the latest available observation even if visually static.
            keep = sorted(set(keep + [len(frames) - 1]))
        kept = frames[keep]
        timestamps = [indices[j] / fps for j in keep]
        assert all(t < end for t in timestamps)
        info = {"decision_id": decisions[i]['decision_id'], "interval_sec": [start, end],
                "frame_indices": indices, "keep_local": keep, "timestamps_sec": timestamps,
                "n_frames_in": len(indices), "n_frames_kept": len(keep),
                "latency_prepare_s": time.perf_counter() - t0}
        prepared.append((kept, info))
        print(f"prepared {i + 1}/{count}: {len(indices)} -> {len(keep)} frames", flush=True)
    del vr
    write_json(args.out_dir / 'inputs.json', [info for _, info in prepared])
    metadata['preparation_wall_s'] = time.perf_counter() - preparation_start
    if args.dry_run:
        metadata['status'] = 'dry_run_complete_no_inference'
        write_json(args.out_dir / 'run.json', metadata)
        print('Dry run complete: no model inference and no evaluation metrics.', flush=True)
        return
    t0 = time.perf_counter()
    judger = QwenVL(args.judger, max_new_tokens=16)
    expert = QwenVL(args.expert, max_new_tokens=128)
    metadata['model_load_wall_s'] = time.perf_counter() - t0
    metadata['device_maps'] = {"judger": {k: str(v) for k, v in judger.model.hf_device_map.items()},
                               "expert": {k: str(v) for k, v in expert.model.hf_device_map.items()}}
    metadata['model_commits'] = {"judger": judger.model.config._commit_hash,
                                 "expert": expert.model.config._commit_hash}
    metadata['status'] = 'running'
    write_json(args.out_dir / 'run.json', metadata)
    summaries = {}
    for mode in dict.fromkeys(args.history_modes):
        history = [{"role": "user", "text": sample['query']}]
        rows = []
        with (args.out_dir / f'{mode}.predictions.jsonl').open('w') as output:
            for i, (frames, info) in enumerate(prepared):
                prior = sample['dialog'][i] if mode == 'gold' else list(history)
                context = context_text(sample['query'], prior, info['timestamps_sec'], info['interval_sec'][1])
                t0 = time.perf_counter()
                result = run_gated(judger, expert, frames, context)
                wall = time.perf_counter() - t0 + info['latency_prepare_s']
                gold = sample['answers'][i]
                if not (gold == '$silent$' or gold.startswith('$interrupt$')):
                    raise ValueError('Unknown gold label')
                row = {**info, **result, "history_mode": mode, "history": prior,
                       "latency_wall_s": wall, "gt_answer": gold,
                       "gt_label": "interrupt" if gold.startswith('$interrupt$') else "silent",
                       "phase": decisions[i]['phase'], "pair_id": decisions[i]['pair_id'],
                       "task_for_audit_only": sample['task'][i]}
                if result['fired']:
                    history.append({"role": "assistant", "text": result['answer']})
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                write_json(args.out_dir / f'{mode}.summary.json', {"complete": len(rows) == count, **summarize(rows)})
                print(f"{mode} {i + 1}/{count} gt={row['gt_label']} pred={row['pred_label']} "
                      f"expert={result['fired']} wall={wall:.2f}s guidance={result['guidance']}", flush=True)
        summaries[mode] = summarize(rows)
    metadata['status'] = 'complete'
    metadata['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(args.out_dir / 'run.json', metadata)
    write_json(args.out_dir / 'summary.json', summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
