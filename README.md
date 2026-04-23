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
- Model: MegaDetector v5a dynamic ONNX (~560 MB, auto-downloaded on first run)
- Model cache: `~/.cache/wildlife-video-labeler/models/md_v5a.0.0-dynamic.onnx`
- No published SHA256 for this model export; a warning is printed on first run. To enable integrity checking, run `sha256sum ~/.cache/wildlife-video-labeler/models/md_v5a.0.0-dynamic.onnx` after download and pass `--model-sha256 <hash>` on future runs.

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
`bear, bobcat, coyote, deer, fox, raccoon, skunk, squirrel` (plus birds: `owl, turkey, woodpecker`)

You can override config/output paths:

```bash
python3 tools/wildlife_video_labeler.py \
  /path/to/trail-camera-drop \
  --label-config config/label-config.json \
  --output-file labels/session-2026-04-22.json
```

## Future work – species classifier

MegaDetector detects the presence of an animal but labels every detection simply `"animal"` — it has no per-species knowledge.  To identify *which* animal is in a clip the standard camera-trap approach is a two-step pipeline:

1. **MegaDetector** (current step) — detects and crops the animal bounding box.
2. **Species classifier** — takes the cropped region and predicts the species.

Candidate classifiers for a northeastern US context:

| Project | Notes |
|---|---|
| [SpeciesNet](https://github.com/google/cameratrap-speciesnet) (Google) | Global-scale; integrated with MegaDetector detections via PyTorchWildlife ≥ 1.2.1 |
| [Deepfaune](https://www.deepfaune.cnrs.fr/en/) | European focus but expanding; "Deepfaune-New-England" variant covers several target species |
| Custom fine-tuned classifier | Train on your own labeled clips (bear, bobcat, coyote, deer, fox, owl, raccoon, skunk, squirrel, turkey, woodpecker) once you have enough verified labels |

None of these classifiers would replace the `MEGADETECTOR_CLASS_NAMES` constants in this repo — they are a separate downstream step that receives the cropped bounding-box image from MegaDetector.
