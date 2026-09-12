"""Reuse LabGate's Qwen loader with actual chat roles and separate visual time streams."""
import time

import torch

from models import QwenVL


class ProactiveQwenVL(QwenVL):
    @torch.inference_mode()
    def generate_window(self, visual, system, text, history):
        content = []
        historical = visual['history_frames']
        for frame, timestamp in zip(historical, visual['history_timestamps_sec']):
            content.extend([{'type': 'text', 'text': f'Historical image at {timestamp:.3f}s:'},
                            {'type': 'image', 'image': frame}])
        content.extend([{'type': 'text', 'text': 'CURRENT video window:'},
                        {'type': 'video', 'video': visual['frames']},
                        {'type': 'text', 'text': text}])
        messages = [{'role': 'system', 'content': system}]
        messages.extend({'role': t['role'], 'content': t['text']} for t in history)
        messages.append({'role': 'user', 'content': content})
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs = {'text': [chat], 'videos': [visual['frames']], 'fps': visual['fps'],
                  'return_tensors': 'pt'}
        if len(historical):
            kwargs['images'] = list(historical)
        inputs = self.processor(**kwargs)
        second_grid = inputs.get('second_per_grid_ts')
        self.last_input_stats = {
            'input_tokens': int(inputs['input_ids'].shape[1]),
            'video_grid_thw': inputs['video_grid_thw'].tolist(),
            'image_grid_thw': inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else [],
            'second_per_grid_ts': second_grid.tolist() if torch.is_tensor(second_grid) else second_grid,
            'fps': visual['fps'], 'current_frames': len(visual['frames']),
            'history_images': len(historical),
        }
        expected = self.processor.image_processor.temporal_patch_size / visual['fps']
        assert all(abs(float(x) - expected) < 1e-6 for x in second_grid)
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        raw = self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        return raw, elapsed
