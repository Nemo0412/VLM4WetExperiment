#!/usr/bin/env python3
"""Fixed eight-decision, 3B-only history/black-frame diagnostic; no prompt tuning."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

from egoproactive_bio import context_text, gate_prompt, parse_gate
from models import QwenVL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run = json.loads((args.run / 'run.json').read_text())
    gold = [json.loads(x) for x in (args.run / 'gold.predictions.jsonl').read_text().splitlines()]
    roll = {r['decision_id']: r for r in
            (json.loads(x) for x in (args.run / 'rollout.predictions.jsonl').read_text().splitlines())}
    selected = {'d02', 'd03', 'd04', 'd05', 'd10', 'd14', 'd18', 'd19'}
    torch.manual_seed(0)
    model = QwenVL(run['args']['judger'], max_new_tokens=16)
    vr = VideoReader(run['video'], ctx=cpu(0), num_threads=2)
    results = []
    contacts = []
    with (args.out / 'predictions.jsonl').open('w') as output:
        for r in gold:
            if r['decision_id'] not in selected:
                continue
            indices = [r['frame_indices'][k] for k in r['keep_local']]
            frames = vr.get_batch(indices).asnumpy()
            base = roll[r['decision_id']]['history']
            clean = [{**t, 'text': t['text'].replace('$interrupt$', '').replace('$silent$', '')}
                     for t in r['history']]
            neutral = {'role': 'assistant', 'text': 'I am observing your task.'}
            tagged = {**neutral, 'text': '$interrupt$' + neutral['text']}
            variants = [
                ('gold_repeat', r['history'], frames),
                ('gold_remove_tags', clean, frames),
                ('rollout_repeat', base, frames),
                ('neutral_history', base + [neutral], frames),
                ('neutral_history_tagged', base + [tagged], frames),
                ('gold_black_frames', r['history'], np.zeros_like(frames)),
                ('rollout_black_frames', base, np.zeros_like(frames)),
            ]
            for name, history, visual in variants:
                prompt = gate_prompt(context_text('Cell passaging', history, r['timestamps_sec'], r['interval_sec'][1]))
                if name == 'gold_repeat':
                    assert prompt == r['judger_prompt']
                if name == 'rollout_repeat':
                    assert prompt == roll[r['decision_id']]['judger_prompt']
                raw, dt = model.generate(visual, prompt)
                item = {'decision_id': r['decision_id'], 'variant': name, 'gt': r['gt_label'],
                        'raw': raw, 'pred': parse_gate(raw), 'generate_s': dt, 'prompt': prompt}
                results.append(item)
                output.write(json.dumps(item) + '\n')
                output.flush()
                print(r['decision_id'], name, repr(raw), flush=True)
            current = [(t, f) for t, f in zip(r['timestamps_sec'], frames) if t >= r['interval_sec'][0]]
            for t, frame in current:
                if r['decision_id'] not in {'d03', 'd05', 'd10', 'd14', 'd19'}:
                    continue
                im = Image.fromarray(frame)
                im.thumbnail((252, 364))
                tile = Image.new('RGB', (280, 405), 'white')
                tile.paste(im, (14, 25))
                ImageDraw.Draw(tile).text((10, 5), f"{r['decision_id']}  t={t:.3f}s", fill='black')
                contacts.append(tile)
    canvas = Image.new('RGB', (280 * 5, 405 * ((len(contacts) + 4) // 5)), '#cccccc')
    for i, im in enumerate(contacts):
        canvas.paste(im, ((i % 5) * 280, (i // 5) * 405))
    canvas.save(args.out / 'current_frames.jpg')
    summary = {name: {'n': len(part := [r for r in results if r['variant'] == name]),
                      'interrupts': sum(r['pred'] == 'interrupt' for r in part),
                      'correct': sum(r['pred'] == r['gt'] for r in part)}
               for name, _, _ in variants}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
