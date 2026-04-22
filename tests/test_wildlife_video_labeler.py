import json
import tempfile
import unittest
from pathlib import Path

from tools.wildlife_video_labeler import (
    find_video_files,
    load_label_config,
    load_labels,
    relative_key,
    save_labels,
)


class WildlifeVideoLabelerTests(unittest.TestCase):
    def test_default_config_includes_bobcat_in_mammals(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = load_label_config(repo_root / "config" / "label-config.json")
        self.assertIn("bobcat", config.get("mammals", []))

    def test_find_video_files_filters_supported_extensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "a.mp4").write_bytes(b"x")
            (base / "b.txt").write_text("not video", encoding="utf-8")
            (base / "nested").mkdir()
            (base / "nested" / "c.MOV").write_bytes(b"x")

            videos = find_video_files(base)

            self.assertEqual(
                [path.relative_to(base).as_posix() for path in videos],
                ["a.mp4", "nested/c.MOV"],
            )

    def test_save_and_load_labels_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            output_file = base / "labels" / "wildlife-video-labels.json"
            labels = {"cam1/clip1.mp4": {"label": "deer", "labeled_at_utc": "2026-01-01T00:00:00+00:00"}}

            save_labels(output_file, base, labels)
            loaded = load_labels(output_file)

            self.assertEqual(loaded, labels)
            raw = json.loads(output_file.read_text(encoding="utf-8"))
            self.assertIn("updated_at_utc", raw)
            self.assertEqual(raw["input_dir"], str(base))

    def test_load_labels_handles_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "labels.json"
            output_file.write_text("", encoding="utf-8")
            self.assertEqual(load_labels(output_file), {})

    def test_relative_key_uses_posix_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            nested = base / "cam1" / "clip1.mp4"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"x")

            self.assertEqual(relative_key(base, nested), "cam1/clip1.mp4")


if __name__ == "__main__":
    unittest.main()
