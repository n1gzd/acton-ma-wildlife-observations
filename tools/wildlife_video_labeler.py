#!/usr/bin/env python3
"""Local-first wildlife video labeling tool for WSL Ubuntu."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".m4v", ".mts", ".mkv"}
DEFAULT_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
)
DEFAULT_MODEL_CACHE_PATH = Path("~/.cache/wildlife-video-labeler/models/yolov8n.onnx")

COCO_PERSON_CLASSES = {0}
COCO_VEHICLE_CLASSES = {1, 2, 3, 5, 7}
COCO_ANIMAL_CLASSES = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def find_video_files(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def load_labels(output_file: Path) -> Dict[str, Dict[str, str]]:
    if not output_file.exists():
        return {}

    with output_file.open("r", encoding="utf-8") as handle:
        raw = handle.read().strip()
        if not raw:
            return {}
        data = json.loads(raw)

    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("Invalid label file: 'labels' must be a dictionary")

    return labels


def load_label_config(config_file: Path) -> Dict[str, List[str]]:
    if not config_file.exists():
        return {}

    with config_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Invalid label config: expected top-level object")

    normalized: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            normalized[key] = [str(item) for item in value]

    return normalized


def save_labels(output_file: Path, input_dir: Path, labels: Dict[str, Dict[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_dir": str(input_dir),
        "labels": labels,
    }
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def open_video(video_path: Path) -> None:
    for command in (["wslview", str(video_path)], ["xdg-open", str(video_path)]):
        try:
            subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue

    print("Could not launch a video player automatically (wslview/xdg-open not found).")


def relative_key(base_dir: Path, path: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def _category_for_coco_class(class_id: int) -> str | None:
    if class_id in COCO_PERSON_CLASSES:
        return "person"
    if class_id in COCO_VEHICLE_CLASSES:
        return "vehicle"
    if class_id in COCO_ANIMAL_CLASSES:
        return "animal"
    return None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_model_file(model_path: Path, model_url: str) -> Path:
    resolved = model_path.expanduser().resolve()
    if resolved.exists():
        return resolved

    resolved.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {resolved} ...")
    urllib.request.urlretrieve(model_url, resolved)  # nosec: B310 - trusted URL input from CLI.
    return resolved


def _probe_video_duration_seconds(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"Non-positive video duration from ffprobe: {video_path}")
    return duration


def _sample_timestamps(duration_seconds: float, frame_count: int) -> List[float]:
    return [duration_seconds * (index + 1) / (frame_count + 1) for index in range(frame_count)]


def _extract_sampled_frames(
    video_path: Path,
    output_dir: Path,
    timestamps: Sequence[float],
    scale_width: int,
) -> List[Path]:
    frame_paths: List[Path] = []
    for index, timestamp in enumerate(timestamps):
        frame_path = output_dir / f"frame-{index:03d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={scale_width}:-1",
                str(frame_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if frame_path.exists():
            frame_paths.append(frame_path)
    return frame_paths


class _YoloOnnxDetector:
    def __init__(self, model_path: Path, input_size: int = 640):
        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        self._np = np
        self._image_module = Image
        self._input_size = input_size
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image_path: Path, conf_threshold: float) -> List[Dict[str, float | int | str]]:
        np = self._np
        image = self._image_module.open(image_path).convert("RGB").resize((self._input_size, self._input_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = np.transpose(array, (2, 0, 1))[np.newaxis, ...]

        output = self._session.run(None, {self._input_name: tensor})[0]
        predictions = np.squeeze(output)
        if predictions.ndim == 3:
            predictions = np.squeeze(predictions, axis=0)
        if predictions.ndim != 2:
            return []
        if predictions.shape[0] in {84, 85} and predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T
        if predictions.shape[1] < 84:
            return []

        boxes_xywh = predictions[:, :4]
        if predictions.shape[1] == 85:
            class_scores = predictions[:, 5:]
            objectness = predictions[:, 4]
            scores = class_scores * objectness[:, np.newaxis]
        else:
            class_scores = predictions[:, 4:]
            scores = class_scores

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        keep_indices = np.where(confidences >= conf_threshold)[0]

        if keep_indices.size == 0:
            return []

        widths = np.abs(boxes_xywh[keep_indices, 2])
        heights = np.abs(boxes_xywh[keep_indices, 3])
        if np.max(widths, initial=0.0) <= 2.0 and np.max(heights, initial=0.0) <= 2.0:
            area_ratios = widths * heights
        else:
            area_ratios = (widths * heights) / float(self._input_size * self._input_size)

        detections: List[Dict[str, float | int | str]] = []
        for output_index, keep_index in enumerate(keep_indices):
            class_id = int(class_ids[keep_index])
            category = _category_for_coco_class(class_id)
            if not category:
                continue
            detections.append(
                {
                    "class_id": class_id,
                    "category": category,
                    "confidence": float(confidences[keep_index]),
                    "area_ratio": float(area_ratios[output_index]),
                }
            )
        return detections


def _build_video_triage_record(
    relative_path: str,
    absolute_path: Path,
    timestamps: Sequence[float],
    detections_by_frame: Sequence[List[Dict[str, float | int | str]]],
    min_frames: int,
    min_box_area_ratio: float,
) -> Dict[str, object]:
    category_summary: Dict[str, Dict[str, float | int | bool]] = {}
    for frame_detections in detections_by_frame:
        frame_categories = set()
        for detection in frame_detections:
            category = str(detection["category"])
            summary = category_summary.setdefault(
                category,
                {"max_confidence": 0.0, "frame_hits": 0, "max_area_ratio": 0.0, "qualified": False},
            )
            summary["max_confidence"] = max(float(summary["max_confidence"]), float(detection["confidence"]))
            summary["max_area_ratio"] = max(float(summary["max_area_ratio"]), float(detection["area_ratio"]))
            if category not in frame_categories:
                summary["frame_hits"] = int(summary["frame_hits"]) + 1
                frame_categories.add(category)

    qualified_categories: List[str] = []
    for category in sorted(category_summary.keys()):
        summary = category_summary[category]
        qualified = int(summary["frame_hits"]) >= min_frames or float(summary["max_area_ratio"]) >= min_box_area_ratio
        summary["qualified"] = qualified
        if qualified:
            qualified_categories.append(category)

    interesting = bool(qualified_categories)
    if "person" in qualified_categories:
        reason = "person_detected"
    elif "animal" in qualified_categories:
        reason = "animal_detected"
    elif "vehicle" in qualified_categories:
        reason = "vehicle_detected"
    else:
        reason = "none_detected"

    return {
        "relative_path": relative_path,
        "absolute_path": str(absolute_path),
        "timestamps_sampled_seconds": [round(value, 3) for value in timestamps],
        "detected_categories_summary": category_summary,
        "qualified_categories": qualified_categories,
        "interesting": interesting,
        "reason": reason,
        "error": None,
    }


def save_triage_reports(
    report_json_path: Path,
    interesting_list_path: Path,
    input_dir: Path,
    records: Sequence[Dict[str, object]],
    settings: Dict[str, object],
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    interesting_list_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_records = sorted(records, key=lambda item: str(item["relative_path"]))
    interesting_paths = [
        str(record["relative_path"]) for record in sorted_records if bool(record.get("interesting", False))
    ]
    error_count = sum(1 for record in sorted_records if record.get("error"))
    payload = {
        "generated_at_utc": _now_utc_iso(),
        "input_dir": str(input_dir),
        "total_videos": len(sorted_records),
        "interesting_videos": len(interesting_paths),
        "errors": error_count,
        "settings": settings,
        "videos": sorted_records,
    }
    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    interesting_list_path.write_text("\n".join(interesting_paths) + ("\n" if interesting_paths else ""), encoding="utf-8")


def run_triage_session(
    input_dir: Path,
    report_json_path: Path,
    interesting_list_path: Path,
    frame_count: int,
    confidence_threshold: float,
    min_frames: int,
    min_box_area_ratio: float,
    model_path: Path,
    model_url: str,
    frame_scale_width: int = 640,
) -> int:
    videos = find_video_files(input_dir)
    total = len(videos)
    if not videos:
        print(f"No video files found under: {input_dir}")
        save_triage_reports(
            report_json_path=report_json_path,
            interesting_list_path=interesting_list_path,
            input_dir=input_dir,
            records=[],
            settings={
                "frame_count": frame_count,
                "confidence_threshold": confidence_threshold,
                "min_frames": min_frames,
                "min_box_area_ratio": min_box_area_ratio,
                "frame_scale_width": frame_scale_width,
                "model_path": str(model_path),
            },
        )
        return 0

    resolved_model_path = _ensure_model_file(model_path, model_url)
    detector = _YoloOnnxDetector(resolved_model_path, input_size=frame_scale_width)

    print(f"Found {total} videos under {input_dir}")
    print("Running CPU triage (non-interactive)...")

    records: List[Dict[str, object]] = []
    interesting_count = 0
    error_count = 0

    for index, video_path in enumerate(videos, start=1):
        rel = relative_key(input_dir, video_path)
        print(f"[{index}/{total}] {rel}")
        try:
            duration = _probe_video_duration_seconds(video_path)
            timestamps = _sample_timestamps(duration, frame_count)
            with tempfile.TemporaryDirectory(prefix="wildlife-triage-") as temp_dir:
                frame_paths = _extract_sampled_frames(
                    video_path=video_path,
                    output_dir=Path(temp_dir),
                    timestamps=timestamps,
                    scale_width=frame_scale_width,
                )
                if not frame_paths:
                    raise RuntimeError("No frames extracted from video")
                detections_by_frame = [
                    detector.detect(frame_path, conf_threshold=confidence_threshold) for frame_path in frame_paths
                ]

            record = _build_video_triage_record(
                relative_path=rel,
                absolute_path=video_path,
                timestamps=timestamps,
                detections_by_frame=detections_by_frame,
                min_frames=min_frames,
                min_box_area_ratio=min_box_area_ratio,
            )
        except Exception as exc:  # pylint: disable=broad-except
            error_count += 1
            record = {
                "relative_path": rel,
                "absolute_path": str(video_path),
                "timestamps_sampled_seconds": [],
                "detected_categories_summary": {},
                "qualified_categories": [],
                "interesting": False,
                "reason": "error",
                "error": str(exc),
            }

        if record.get("interesting", False):
            interesting_count += 1
        records.append(record)

    settings = {
        "frame_count": frame_count,
        "confidence_threshold": confidence_threshold,
        "min_frames": min_frames,
        "min_box_area_ratio": min_box_area_ratio,
        "frame_scale_width": frame_scale_width,
        "model_path": str(resolved_model_path),
        "model_url": model_url,
    }
    save_triage_reports(
        report_json_path=report_json_path,
        interesting_list_path=interesting_list_path,
        input_dir=input_dir,
        records=records,
        settings=settings,
    )

    print(f"Triage complete. total={total} interesting={interesting_count} errors={error_count}")
    print(f"JSON report: {report_json_path}")
    print(f"Interesting list: {interesting_list_path}")
    return 0


def run_labeling_session(
    input_dir: Path,
    output_file: Path,
    default_label: str = "",
    label_config: Dict[str, List[str]] | None = None,
) -> int:
    videos = find_video_files(input_dir)
    if not videos:
        print(f"No video files found under: {input_dir}")
        return 0

    labels = load_labels(output_file)
    index = 0

    label_config = label_config or {}
    mammal_labels = label_config.get("mammals", [])
    mammal_labels_text = ", ".join(mammal_labels) if mammal_labels else "none configured"

    help_text = (
        "Commands: <label text>, /skip, /back, /open, /quit, /help\n"
        "- Enter plain text to set/update label for current video.\n"
        "- /skip advances without changing current label."
    )

    print(f"Found {len(videos)} video files under {input_dir}")
    print(f"Mammal labels from config: {mammal_labels_text}")
    print(help_text)

    while 0 <= index < len(videos):
        current_path = videos[index]
        key = relative_key(input_dir, current_path)
        existing = labels.get(key, {}).get("label", "")
        shown_label = existing if existing else "<unlabeled>"

        print(f"\n[{index + 1}/{len(videos)}] {key}")
        print(f"Current label: {shown_label}")
        entry = input("Label or command: ").strip()

        if not entry:
            entry = default_label

        if entry == "/help":
            print(help_text)
            continue

        if entry == "/quit":
            break

        if entry == "/open":
            open_video(current_path)
            continue

        if entry == "/back":
            index = max(index - 1, 0)
            continue

        if entry == "/skip":
            index += 1
            continue

        labels[key] = {
            "label": entry,
            "labeled_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        save_labels(output_file, input_dir, labels)
        index += 1

    save_labels(output_file, input_dir, labels)
    labeled_count = sum(1 for value in labels.values() if value.get("label"))
    print(f"Saved {labeled_count} labeled entries to {output_file}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-first wildlife video labeling tool")
    parser.add_argument(
        "mode",
        nargs="?",
        default="label",
        help="Mode: 'label' (default) or 'triage'. For triage subcommand use: triage <input_dir>",
    )
    parser.add_argument("input_dir", nargs="?", type=Path, help="Directory containing raw trail camera videos")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("labels/wildlife-video-labels.json"),
        help="Label mode JSON output (default: labels/wildlife-video-labels.json)",
    )
    parser.add_argument(
        "--default-label",
        default="",
        help="Label mode default label to apply when pressing enter on an empty prompt",
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=Path("config/label-config.json"),
        help="Label mode config JSON (default: config/label-config.json)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/wildlife-video-triage.json"),
        help="Triage mode JSON report path (default: reports/wildlife-video-triage.json)",
    )
    parser.add_argument(
        "--interesting-list",
        type=Path,
        default=Path("reports/interesting-videos.txt"),
        help="Triage mode text file with relative paths of interesting videos",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Triage mode number of evenly spaced frames sampled per video (default: 6)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Triage mode confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=2,
        help="Triage mode minimum sampled frames containing category to qualify (default: 2)",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=0.02,
        help="Triage mode minimum normalized box area to qualify even if seen in fewer frames (default: 0.02)",
    )
    parser.add_argument(
        "--model-url",
        default=DEFAULT_MODEL_URL,
        help="Triage mode model URL used for first-run download",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_CACHE_PATH,
        help="Triage mode local ONNX model path (default: ~/.cache/wildlife-video-labeler/models/yolov8n.onnx)",
    )
    args = parser.parse_args()

    if args.mode != "triage" and args.input_dir is None:
        args.input_dir = Path(args.mode)
        args.mode = "label"

    if args.input_dir is None:
        parser.error("input_dir is required")
    return args


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return 2

    if args.mode == "triage":
        if args.frames <= 0:
            print("--frames must be > 0")
            return 2
        if args.conf <= 0 or args.conf > 1:
            print("--conf must be in (0, 1]")
            return 2
        if args.min_frames <= 0:
            print("--min-frames must be > 0")
            return 2
        if args.min_box_area <= 0:
            print("--min-box-area must be > 0")
            return 2

        return run_triage_session(
            input_dir=input_dir,
            report_json_path=args.report_json.expanduser().resolve(),
            interesting_list_path=args.interesting_list.expanduser().resolve(),
            frame_count=args.frames,
            confidence_threshold=args.conf,
            min_frames=args.min_frames,
            min_box_area_ratio=args.min_box_area,
            model_path=args.model_path,
            model_url=args.model_url,
        )

    output_file = args.output_file.expanduser().resolve()
    label_config_file = args.label_config.expanduser().resolve()
    label_config = load_label_config(label_config_file)
    return run_labeling_session(
        input_dir=input_dir,
        output_file=output_file,
        default_label=args.default_label,
        label_config=label_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
