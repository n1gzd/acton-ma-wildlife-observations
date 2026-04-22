# acton-ma-wildlife-observations

Learning to use AI to observe wildlife in Acton MA.

## Setup (WSL Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
sudo apt-get update && sudo apt-get install -y ffmpeg
pip install -r requirements.txt
```

## Video triage mode (non-interactive, CPU-only)

Triage scans videos and writes reports only (no move/copy/symlink of video files).

```bash
python3 tools/wildlife_video_labeler.py triage /path/to/trail-camera-drop
```

Defaults:
- JSON report: `reports/wildlife-video-triage.json`
- Interesting list: `reports/interesting-videos.txt`
- Sampled frames per video: `--frames 6`
- Confidence threshold: `--conf 0.35`
- Wind/weather filter: `--min-frames 2` OR `--min-box-area 0.02`
- Model cache: `~/.cache/wildlife-video-labeler/models/yolov8n.onnx` (auto-downloaded on first run)

Example with overrides:

```bash
python3 tools/wildlife_video_labeler.py triage /path/to/trail-camera-drop \
  --report-json reports/session-triage.json \
  --interesting-list reports/session-interesting.txt \
  --frames 8 \
  --conf 0.4 \
  --min-frames 3
```

## Interactive labeling mode (existing workflow)

Use the local CLI labeler to walk video files and save labels in JSON on disk:

```bash
python3 tools/wildlife_video_labeler.py /path/to/trail-camera-drop
```

- Default output: `labels/wildlife-video-labels.json`
- Default label config: `config/label-config.json`

The default mammal label list now includes:
`bear, bobcat, coyote, deer, fox, raccoon, skunk, squirrel`

You can override config/output paths:

```bash
python3 tools/wildlife_video_labeler.py \
  /path/to/trail-camera-drop \
  --label-config config/label-config.json \
  --output-file labels/session-2026-04-22.json
```
