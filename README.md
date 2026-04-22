# acton-ma-wildlife-observations

Learning to use AI to observe wildlife in Acton MA.

## Local-first wildlife video labeling (WSL Ubuntu, Python)

Use the local CLI labeler to walk video files and save labels in JSON on disk:

```bash
python tools/wildlife_video_labeler.py /path/to/trail-camera-drop
```

- Default output: `labels/wildlife-video-labels.json`
- Default label config: `config/label-config.json`

The default mammal label list now includes:
`bear, bobcat, coyote, deer, fox, raccoon, skunk, squirrel`

You can override config/output paths:

```bash
python tools/wildlife_video_labeler.py \
  /path/to/trail-camera-drop \
  --label-config config/label-config.json \
  --output-file labels/session-2026-04-22.json
```
