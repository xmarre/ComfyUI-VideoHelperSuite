import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from videohelpersuite.audio_mux import (
    build_audio_mux_args,
    mux_audio_with_sigfpe_fallback,
)


class BuildAudioMuxArgsTests(unittest.TestCase):
    def test_uses_pcm16_without_apad_and_limits_audio_threads(self):
        args = build_audio_mux_args(
            "/usr/bin/ffmpeg",
            "/tmp/video.mp4",
            "/tmp/output.mp4",
            32000,
            2,
            ["-c:a", "aac"],
        )

        self.assertIn("s16le", args)
        self.assertNotIn("f32le", args)
        self.assertFalse(any("apad" in arg for arg in args))
        self.assertEqual(args[args.index("-threads:a") + 1], "1")

    def test_scalar_fallback_disables_cpu_flags(self):
        args = build_audio_mux_args(
            "/usr/bin/ffmpeg",
            "/tmp/video.mp4",
            "/tmp/output.mp4",
            32000,
            2,
            ["-c:a", "aac"],
            disable_cpu_flags=True,
        )

        self.assertEqual(args[args.index("-cpuflags") + 1], "0")


class AudioMuxFallbackTests(unittest.TestCase):
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_retries_sigfpe_with_cpu_simd_disabled(self, run_mock):
        run_mock.side_effect = [
            subprocess.CompletedProcess([], -signal.SIGFPE, b"", b"primary"),
            subprocess.CompletedProcess([], 0, b"", b"fallback"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            result = mux_audio_with_sigfpe_fallback(
                ffmpeg_path="/usr/bin/ffmpeg",
                video_path="/tmp/video.mp4",
                output_path=output_path,
                sample_rate=32000,
                channels=2,
                audio_pass=["-c:a", "aac"],
                audio_data=b"pcm",
                env={},
            )

        self.assertTrue(result.used_scalar_fallback)
        self.assertEqual(run_mock.call_count, 2)
        primary_args = run_mock.call_args_list[0].args[0]
        fallback_args = run_mock.call_args_list[1].args[0]
        self.assertNotIn("-cpuflags", primary_args)
        self.assertEqual(
            fallback_args[fallback_args.index("-cpuflags") + 1],
            "0",
        )


if __name__ == "__main__":
    unittest.main()
