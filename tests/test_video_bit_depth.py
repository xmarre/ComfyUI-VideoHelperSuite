import copy
import json
import unittest
from pathlib import Path

from videohelpersuite.video_bit_depth import (
    apply_video_bit_depth,
    normalize_video_bit_depth,
)


FORMATS_DIR = Path(__file__).resolve().parents[1] / "video_formats"
BIT_DEPTH_FORMATS = {
    "h264-mp4.json": (8, "yuv420p", "yuv420p10le"),
    "h264-mp4-wsl-safe.json": (8, "yuv420p", "yuv420p10le"),
    "h265-mp4.json": (10, "yuv420p", "yuv420p10le"),
    "h265-mp4-wsl-safe.json": (10, "yuv420p", "yuv420p10le"),
    "av1-webm.json": (10, "yuv420p", "yuv420p10le"),
    "nvenc_h264-mp4.json": (8, "yuv420p", "p010le"),
    "nvenc_hevc-mp4.json": (8, "yuv420p", "p010le"),
    "nvenc_av1-mp4.json": (8, "yuv420p", "p010le"),
}


def output_pixel_format(video_format):
    index = video_format["main_pass"].index("-pix_fmt")
    return video_format["main_pass"][index + 1]


class VideoBitDepthTests(unittest.TestCase):
    def test_normalizes_widget_and_manual_values(self):
        self.assertEqual(normalize_video_bit_depth(8), 8)
        self.assertEqual(normalize_video_bit_depth(8.0), 8)
        self.assertEqual(normalize_video_bit_depth("10"), 10)
        self.assertEqual(normalize_video_bit_depth("10-bit"), 10)
        self.assertEqual(normalize_video_bit_depth("8bit"), 8)

    def test_rejects_invalid_depths(self):
        for value in (None, True, 8.5, 9, 12, "auto", ""):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "8 or 10"):
                    normalize_video_bit_depth(value)

    def test_builtin_formats_map_output_and_pipe_depth_together(self):
        for filename, (default, expected_8, expected_10) in BIT_DEPTH_FORMATS.items():
            with self.subTest(filename=filename):
                with (FORMATS_DIR / filename).open(encoding="utf-8") as stream:
                    source = json.load(stream)

                widgets = {
                    definition[0]: definition
                    for definition in source.get("extra_widgets", [])
                }
                self.assertIn("bit_depth", widgets)
                self.assertEqual(widgets["bit_depth"][2]["default"], default)

                configured_8 = apply_video_bit_depth(copy.deepcopy(source), 8)
                self.assertEqual(output_pixel_format(configured_8), expected_8)
                self.assertEqual(configured_8["input_color_depth"], "8bit")

                configured_10 = apply_video_bit_depth(copy.deepcopy(source), 10)
                self.assertEqual(output_pixel_format(configured_10), expected_10)
                self.assertEqual(configured_10["input_color_depth"], "16bit")

    def test_non_opted_in_format_is_unchanged(self):
        source = {"main_pass": ["-pix_fmt", "yuv420p"]}
        self.assertIs(apply_video_bit_depth(source, None), source)
        self.assertEqual(source["main_pass"], ["-pix_fmt", "yuv420p"])

    def test_fixed_high_bit_depth_prores_uses_high_precision_input(self):
        with (FORMATS_DIR / "ProRes.json").open(encoding="utf-8") as stream:
            source = json.load(stream)
        self.assertEqual(source["input_color_depth"], "16bit")

    def test_malformed_opted_in_format_fails_before_encoding(self):
        source = {"bit_depths": {"8": "yuv420p", "10": "yuv420p10le"}, "main_pass": []}
        with self.assertRaisesRegex(ValueError, "missing a -pix_fmt"):
            apply_video_bit_depth(source, 10)


if __name__ == "__main__":
    unittest.main()
