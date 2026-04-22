#!/usr/bin/env python3
"""Local-first wildlife video labeling tool for WSL Ubuntu."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".m4v", ".mts", ".mkv"}


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
    parser.add_argument("input_dir", type=Path, help="Directory containing raw trail camera videos")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("labels/wildlife-video-labels.json"),
        help="Local JSON file for labels (default: labels/wildlife-video-labels.json)",
    )
    parser.add_argument(
        "--default-label",
        default="",
        help="Optional label to apply when pressing enter on an empty prompt",
    )
    parser.add_argument(
        "--label-config",
        type=Path,
        default=Path("config/label-config.json"),
        help="Label config JSON (default: config/label-config.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    label_config_file = args.label_config.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return 2

    label_config = load_label_config(label_config_file)
    return run_labeling_session(
        input_dir=input_dir,
        output_file=output_file,
        default_label=args.default_label,
        label_config=label_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
