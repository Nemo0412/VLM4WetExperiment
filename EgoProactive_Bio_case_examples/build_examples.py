#!/usr/bin/env python3
"""Build two reviewable EgoProactive-Bio E2E case bundles.

The generated clips are for human inspection. The contact sheets and kept-frame
PNGs identify the 16 sampled frames and the exact local indices recorded by the
reprojection evaluation artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXP = REPO / "LabGate/experiments/egoproactive_bio_contextual_20260906"
DATA = EXP / "data_official"

CASES = {
    "success_d10": {
        "id": "d10",
        "outcome": "success",
        "summary_zh": (
            "3B 正确触发 action_error；32B 正确指出培养瓶在移除培养基后不能干置，"
            "并要求立即使用 PBS 冲洗。"
        ),
    },
    "failure_d14": {
        "id": "d14",
        "outcome": "failure",
        "summary_zh": (
            "标注要求对加入胰酶后的剧烈摇晃发出 action_error，但 3B 输出 NO，"
            "使 32B 在真实 E2E 流程中没有被调用。"
        ),
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def get_font(size: int):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_frames(video: Path, frame_indices: list[int]) -> tuple[list, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {index} from {video}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def write_contact_sheet(
    path: Path, frames: list, source_indices: list[int], fps: float
) -> None:
    thumb_w = 240
    first_h, first_w = frames[0].shape[:2]
    thumb_h = round(first_h * thumb_w / first_w)
    label_h = 34
    cols, rows = 4, 4
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(16)
    for local, (frame, source_index) in enumerate(zip(frames, source_indices)):
        image = Image.fromarray(frame).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (local % cols) * thumb_w
        y = (local // cols) * (thumb_h + label_h)
        canvas.paste(image, (x, y))
        draw.text(
            (x + 7, y + thumb_h + 7),
            f"local {local:02d} | frame {source_index} | {source_index / fps:.3f}s",
            fill="black",
            font=font,
        )
    canvas.save(path, quality=92)


def write_clip(path: Path, video: Path, interval: list[float], max_side: int = 720) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_side / max(width, height))
    out_w = max(2, round(width * scale) // 2 * 2)
    out_h = max(2, round(height * scale) // 2 * 2)
    output_fps = min(source_fps, 15.0)
    sample_every = max(1, round(source_fps / output_fps))
    start_frame = round(interval[0] * source_fps)
    end_frame = round(interval[1] * source_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (out_w, out_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    written = 0
    current = start_frame
    while current <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (current - start_frame) % sample_every == 0:
            if (out_w, out_h) != (width, height):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
        current += 1
    writer.release()
    cap.release()
    return {
        "source_interval_sec": interval,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "output_size": [out_w, out_h],
        "output_frames": written,
        "note": "This clip is for human inspection; model input is shown by the frame artifacts.",
    }


def main() -> None:
    contextual = read_jsonl(DATA / "gate_contextual.jsonl")
    expert_data = read_jsonl(DATA / "expert_contextual.jsonl")
    cascade = read_jsonl(EXP / "cascade/predictions.jsonl")
    oracle = read_jsonl(EXP / "expert_oracle/predictions.jsonl")

    for folder_name, spec in CASES.items():
        case_dir = HERE / folder_name
        case_dir.mkdir(parents=True, exist_ok=True)
        decision_id = spec["id"]
        data = next(row for row in contextual if row["id"] == decision_id)
        expert_label = next(row for row in expert_data if row["id"] == decision_id)
        e2e = next(
            row
            for row in cascade
            if row["id"] == decision_id
            and row["variant"] == "official_raw_reprojection"
        )
        forced = next(
            row
            for row in oracle
            if row["id"] == decision_id
            and row["variant"] == "official_raw_reprojection"
        )

        source_video = Path(data["video"])
        frames, fps = read_frames(source_video, data["frame_indices"])
        write_contact_sheet(
            case_dir / "sampled_16_frames.jpg", frames, data["frame_indices"], fps
        )
        kept_source_indices = []
        for local_index in e2e["keep_local"]:
            source_index = data["frame_indices"][local_index]
            kept_source_indices.append(source_index)
            Image.fromarray(frames[local_index]).save(
                case_dir / f"model_kept_local_{local_index:02d}_source_{source_index}.png"
            )
        clip_metadata = write_clip(
            case_dir / "context_clip.mp4", source_video, data["interval_sec"]
        )

        bundle = {
            "case_id": decision_id,
            "outcome": spec["outcome"],
            "variant": "official_raw_reprojection",
            "summary_zh": spec["summary_zh"],
            "source": {
                "dataset_video": str(source_video),
                "annotated_interval_sec": data["interval_sec"],
                "sampled_source_frame_indices": data["frame_indices"],
                "reprojection_keep_local_indices": e2e["keep_local"],
                "reprojection_kept_source_indices": kept_source_indices,
                "n_frames_original": e2e["n_frames_original"],
                "n_frames_sent": e2e["n_frames_sent"],
                "clip_metadata": clip_metadata,
            },
            "model_context": {
                "history": e2e["history_sent_to_both"],
                "current_step": e2e["current_step_sent_to_both"],
                "asr_text": data["asr_text"],
            },
            "ground_truth": {
                "fire": data["gt_fire"],
                "reason": data["gt_reason"],
                "type": data["gt_type"],
                "reference_guidance": expert_label["reference_guidance"],
            },
            "actual_e2e": {
                "judger_prompt_file": "judger_prompt.txt",
                "judger_raw": e2e["judger_raw"],
                "parsed_fire": e2e["fired"],
                "parsed_reason": e2e["reason"],
                "judger_latency_s": e2e["latency_judger_s"],
                "expert_called": bool(e2e["fired"]),
                "expert_prompt_file": "expert_prompt.txt" if e2e["fired"] else None,
                "expert_raw": e2e["expert_raw"],
                "predicted_type": e2e["pred_type"],
                "message": e2e["message"],
                "expert_latency_s": e2e["latency_expert_s"],
            },
            "forced_oracle_expert_for_diagnosis": {
                "warning": (
                    "Diagnostic decoupled result with the gold trigger reason; "
                    "it is not the actual E2E output."
                ),
                "prompt_file": "oracle_expert_prompt.txt",
                "oracle_reason_sent": forced["oracle_reason_sent"],
                "raw": forced["raw"],
                "predicted_type": forced["pred_type"],
                "message": forced["message"],
                "type_correct": forced["type_correct"],
                "semantic": forced["semantic"],
            },
        }
        (case_dir / "case.json").write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (case_dir / "history.json").write_text(
            json.dumps(e2e["history_sent_to_both"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (case_dir / "judger_prompt.txt").write_text(e2e["judger_prompt"], encoding="utf-8")
        (case_dir / "judger_output.txt").write_text(e2e["judger_raw"] + "\n", encoding="utf-8")
        if e2e["fired"]:
            (case_dir / "expert_prompt.txt").write_text(e2e["expert_prompt"], encoding="utf-8")
            (case_dir / "expert_output.txt").write_text(e2e["expert_raw"] + "\n", encoding="utf-8")
        (case_dir / "oracle_expert_prompt.txt").write_text(forced["prompt"], encoding="utf-8")
        (case_dir / "oracle_expert_output.txt").write_text(forced["raw"] + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
