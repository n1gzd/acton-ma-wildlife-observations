import json
import tempfile
import unittest
from pathlib import Path

from tools.wildlife_video_labeler import (
    find_video_files,
    load_label_config,
    load_labels,
    parse_args,
    relative_key,
    save_labels,
    save_triage_reports,
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

    def test_parse_args_keeps_legacy_label_mode(self):
        args = parse_args(["/tmp/input"])
        self.assertEqual(args.mode, "label")
        self.assertEqual(args.input_dir, Path("/tmp/input"))

    def test_parse_args_supports_explicit_label_subcommand(self):
        args = parse_args(["label", "/tmp/input"])
        self.assertEqual(args.mode, "label")
        self.assertEqual(args.input_dir, Path("/tmp/input"))

    def test_parse_args_supports_triage_subcommand(self):
        args = parse_args(["triage", "/tmp/input", "--frames", "4"])
        self.assertEqual(args.mode, "triage")
        self.assertEqual(args.input_dir, Path("/tmp/input"))
        self.assertEqual(args.frames, 4)

    def test_save_triage_reports_writes_sorted_interesting_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            report_json = base / "reports" / "triage.json"
            interesting_list = base / "reports" / "interesting.txt"

            records = [
                {
                    "relative_path": "z/clip3.mp4",
                    "absolute_path": "/abs/z/clip3.mp4",
                    "timestamps_sampled_seconds": [1.0],
                    "detected_categories_summary": {},
                    "qualified_categories": [],
                    "interesting": False,
                    "reason": "none_detected",
                    "error": None,
                },
                {
                    "relative_path": "a/clip1.mp4",
                    "absolute_path": "/abs/a/clip1.mp4",
                    "timestamps_sampled_seconds": [1.0],
                    "detected_categories_summary": {"animal": {"max_confidence": 0.9}},
                    "qualified_categories": ["animal"],
                    "interesting": True,
                    "reason": "animal_detected",
                    "error": None,
                },
            ]

            save_triage_reports(
                report_json_path=report_json,
                interesting_list_path=interesting_list,
                input_dir=base,
                records=records,
                settings={"frames": 6},
            )

            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_videos"], 2)
            self.assertEqual(payload["interesting_videos"], 1)
            self.assertEqual([video["relative_path"] for video in payload["videos"]], ["a/clip1.mp4", "z/clip3.mp4"])
            self.assertEqual(interesting_list.read_text(encoding="utf-8"), "a/clip1.mp4\n")


if __name__ == "__main__":
    unittest.main()
