"""Structured small-model gate, causal state, and reason-conditioned expert prompts."""
from __future__ import annotations

import json
import math
import re

from egoproactive_bio import CELL_PASSAGING_PROTOCOL, parse_gate
from prompts import trigger_context

JUDGER_SYSTEM = """You are the small visual gate for a proactive procedural assistant.
Base the current decision on the CURRENT video, the supplied SOP, and previous
observations. Earlier images and dialogue are historical context, not current evidence.
Previous model observations may be wrong: correct them when the current video disagrees.
Classify why an interruption is useful:
- safety_warning: a currently visible unsafe condition needs an immediate warning.
- action_error: the currently observed action conflicts with the supplied protocol.
- next_step: a step has just completed and its next action has not already been explained.
- user_query: the user has actually asked a new question.
- none: ongoing correct work, following a correction, already-explained next action, or task finished.
Do not infer elapsed process time from absolute video time or edited cuts alone.
Do not infer a missing step just because it was absent from sparse historical images.
Return ONLY one JSON object with these five keys:
1. decision: choose exactly one string, "interrupt" or "silent".
2. reason_type: choose one of the five categories defined above.
3. observation: one short clause describing evidence in the current video.
4. current_step: the action you currently observe, or "unknown".
5. step_status: choose one string from "in_progress", "completed", "correcting", "unknown".
For silent use reason_type=none. For interrupt choose the best supported non-none reason.
Choose a single value for each field; never concatenate alternatives. Do not generate coaching sentences."""

EXPERT_SYSTEM = """You are the large visual guidance model in a proactive procedural assistant.
The small model has requested an interruption and supplies its reason, observation,
and estimated step. These are hypotheses, not verified facts. Check them against the
CURRENT video and SOP. Older images, prior guidance and previous observations are context.
Give one or two concise English sentences addressing the current supported safety issue,
action error, or useful next step. Do not merely repeat the small model's reason.
If the claimed hazard is unsupported, do not assert it; give a brief conditional check
or ask for clarification rather than inventing an error. Do not introduce unsupported
quantities or repeat resolved warnings. Output only the spoken guidance, without
decision labels, reasoning traces, or JSON."""


def clean_history(history, max_turns=4):
    """Keep actual chat roles and spoken content, removing control tags from history."""
    turns = []
    for turn in history:
        text = re.sub(r"\$(?:interrupt|silent)\$", "", turn.get('text', ''), flags=re.I).strip()
        if text:
            turns.append({'role': turn['role'] if turn['role'] in {'user', 'assistant'} else 'user',
                          'text': text})
    # The evaluator always starts histories with the task query.
    opening = turns[:1]
    rest = turns[1:]
    return opening + (rest[-max_turns:] if max_turns > 0 else [])


def regular_window(start, end, source_fps, n_video, target_fps=2.0, max_frames=32):
    """An even-length, constant-stride current clip ending before the decision time."""
    if not (0 <= start < end and source_fps > 0 and target_fps > 0 and n_video > 0 and max_frames >= 2):
        raise ValueError('Invalid sampling configuration')
    lo = math.ceil(start * source_fps)
    hi = min(n_video - 1, math.ceil(end * source_fps) - 1)
    if hi - lo < 1:
        raise ValueError('At least two source frames are required')
    stride = max(1, round(source_fps / target_fps))
    stride = min(stride, hi - lo)
    # Cap long clips without deleting irregularly spaced interior frames.
    stride = max(stride, math.ceil((hi - lo) / (max_frames - 1)))
    indices = list(range(hi, lo - 1, -stride))[::-1]
    if len(indices) % 2:
        indices = indices[1:]
    if len(indices) < 2:
        raise ValueError('Sampling produced too few frames')
    return indices, source_fps / stride


def historical_indices(start, source_fps, count=4):
    if count <= 0:
        return []
    last = math.ceil(start * source_fps) - 1
    if last < 0:
        return []
    n = min(count, last + 1)
    return sorted({round(i * last / (n - 1)) for i in range(n)}) if n > 1 else [last]


def parse_structured_gate(raw):
    text = (raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
        text = re.sub(r'\s*```$', '', text)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if not isinstance(obj, dict):
        # Compatibility fallback is recorded; it does not invent visual evidence.
        legacy = re.fullmatch(r'\s*(\$interrupt\$|\$silent\$|yes|no|interrupt|interupt|silent)[.!]?\s*', raw or '', re.I)
        return {'decision': parse_gate(legacy.group(1)) if legacy else 'invalid', 'reason_type': 'unknown', 'observation': '',
                'current_step': 'unknown', 'step_status': 'unknown', 'schema_valid': False}
    # A structured enum must be exact. "interrupt|silent" is NOT an interrupt.
    raw_decision = obj.get('decision')
    normalized = raw_decision.strip().lower() if isinstance(raw_decision, str) else ''
    decision = ({'interrupt': 'interrupt', 'interupt': 'interrupt', 'yes': 'interrupt', '$interrupt$': 'interrupt',
                 'silent': 'silent', 'no': 'silent', '$silent$': 'silent'}).get(normalized, 'invalid')
    reason = str(obj.get('reason_type', 'unknown')).strip().lower().replace(' ', '_')
    reason = {'safety': 'safety_warning', 'assistant': 'next_step'}.get(reason, reason)
    allowed = {'safety_warning', 'action_error', 'next_step', 'user_query', 'none'}
    obs = obj.get('observation', '')
    step = obj.get('current_step', 'unknown')
    raw_status = obj.get('step_status', 'unknown')
    status = raw_status if isinstance(raw_status, str) else 'unknown'
    valid = (decision in {'interrupt', 'silent'} and reason in allowed
             and isinstance(obs, str) and bool(obs.strip()) and isinstance(step, str)
             and isinstance(raw_status, str) and status in {'in_progress', 'completed', 'correcting', 'unknown'}
             and ((decision == 'silent' and reason == 'none')
                  or (decision == 'interrupt' and reason != 'none')))
    return {'decision': decision, 'reason_type': reason if reason in allowed else 'unknown',
            'observation': obs[:500] if isinstance(obs, str) else '',
            'current_step': step[:200] if isinstance(step, str) else 'unknown',
            'step_status': status if status in {'in_progress', 'completed', 'correcting'} else 'unknown',
            'schema_valid': valid}


def user_context(query, visual, previous_state):
    return (
        f'TASK: {query}\nPROTOCOL:\n{CELL_PASSAGING_PROTOCOL}\n'
        f'CURRENT WINDOW: {visual["interval_sec"]} seconds. '
        f'Current video is regularly sampled at {visual["fps"]:.6f} fps.\n'
        'Current frame timestamps (seconds): ' + json.dumps(visual['timestamps_sec']) + '\n'
        'Historical image timestamps (seconds): ' + json.dumps(visual['history_timestamps_sec']) + '\n'
        'Previous model observations (unverified, from earlier decisions only): '
        + json.dumps(previous_state[-2:], ensure_ascii=False) + '\n'
        'No audio track or new user question is available. Judge the current window end.\n'
    )


def run_step(judger, expert, visual, query, history, previous_state):
    clean = clean_history(history)
    text = user_context(query, visual, previous_state)
    raw, jt = judger.generate_window(visual, JUDGER_SYSTEM, text, clean)
    parsed = parse_structured_gate(raw)
    jstats = dict(getattr(judger, 'last_input_stats', {}))
    fired = parsed['decision'] == 'interrupt'
    e_text = ''
    expert_raw, et, estats = '', 0.0, {}
    if fired:
        e_text = text + trigger_context(parsed['reason_type'], parsed['observation'], parsed['current_step'])
        expert_raw, et = expert.generate_window(visual, EXPERT_SYSTEM, e_text, clean)
        estats = dict(getattr(expert, 'last_input_stats', {}))
    guidance = re.sub(r'^\s*\$interrupt\$\s*', '', expert_raw).strip()
    return {'pred_label': parsed['decision'], 'fired': fired, 'trigger': parsed,
            'judger_raw': raw, 'expert_raw': expert_raw, 'guidance': guidance,
            'guidance_nonempty': fired and bool(guidance) and parse_gate(guidance) != 'silent',
            'answer': '$interrupt$' + guidance if fired else '$' + parsed['decision'] + '$',
            'history_sent': clean, 'previous_state': list(previous_state[-2:]),
            'judger_system_prompt': JUDGER_SYSTEM, 'judger_user_prompt': text,
            'expert_system_prompt': EXPERT_SYSTEM if fired else '', 'expert_user_prompt': e_text,
            'judger_input_stats': jstats, 'expert_input_stats': estats,
            'latency_judger_generate_s': jt, 'latency_expert_generate_s': et}


def update_state(previous_state, result, decision_time):
    """Track model observations even on silent decisions, without gold steps or labels."""
    p = result['trigger']
    if p['schema_valid']:
        previous_state.append({'time_sec': decision_time, 'current_step': p['current_step'],
                               'step_status': p['step_status'], 'observation': p['observation']})
        del previous_state[:-2]
