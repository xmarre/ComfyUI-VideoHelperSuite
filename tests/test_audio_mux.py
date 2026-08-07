import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from videohelpersuite.audio_mux import (
    AudioMuxError,
    DEFAULT_WSL_FINALIZE_TIMEOUT,
    build_audio_mux_args,
    configured_finalize_timeout,
    mux_audio_with_sigfpe_fallback,
)


class AudioMuxCommandTests(unittest.TestCase):
    def test_builds_explicit_pcm_input_and_stream_mapping(self):
        args = build_audio_mux_args(
            "/usr/bin/ffmpeg",
            "video.mp4",
            "output.mp4",
            32000,
            2,
            ["-c:a", "aac"],
        )

        self.assertIn("s16le", args)
        self.assertNotIn("f32le", args)
        self.assertFalse(any("apad" in arg for arg in args))
        self.assertEqual(args[args.index("-threads:a") + 1], "1")

        pcm_input = len(args) - 1 - args[::-1].index("-i")
        self.assertEqual(args[pcm_input + 1], "-")
        self.assertEqual(
            args[pcm_input - 6 : pcm_input],
            ["-ar", "32000", "-ac", "2", "-f", "s16le"],
        )
        self.assertEqual(
            args[args.index("-map") : args.index("-c:v")],
            ["-map", "0:v:0", "-map", "1:a:0"],
        )


class AudioMuxTimeoutTests(unittest.TestCase):
    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=True)
    def test_wsl_has_bounded_default_timeout(self, _is_wsl_mock):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                configured_finalize_timeout(),
                DEFAULT_WSL_FINALIZE_TIMEOUT,
            )

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=True)
    def test_explicit_zero_disables_wsl_default_timeout(self, _is_wsl_mock):
        with mock.patch.dict(
            os.environ,
            {"VHS_FFMPEG_FINALIZE_TIMEOUT": "0"},
            clear=True,
        ):
            self.assertIsNone(configured_finalize_timeout())


class AudioMuxFallbackTests(unittest.TestCase):
    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=False)
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_sigfpe_retries_without_cpu_flags(self, run_mock, _is_wsl_mock):
        run_mock.side_effect = [
            subprocess.CompletedProcess([], -signal.SIGFPE, b"", b"primary"),
            subprocess.CompletedProcess([], 0, b"", b"fallback"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = mux_audio_with_sigfpe_fallback(
                ffmpeg_path="/usr/bin/ffmpeg",
                video_path="video.mp4",
                output_path=Path(temp_dir) / "output.mp4",
                sample_rate=32000,
                channels=2,
                audio_pass=["-c:a", "aac"],
                audio_data=b"pcm",
                env={},
            )

        self.assertTrue(result.used_scalar_fallback)
        self.assertEqual(result.stderr, "fallback")
        self.assertEqual(run_mock.call_count, 2)
        primary_args = run_mock.call_args_list[0].args[0]
        fallback_args = run_mock.call_args_list[1].args[0]
        self.assertNotIn("-cpuflags", primary_args)
        self.assertEqual(
            fallback_args[fallback_args.index("-cpuflags") + 1],
            "0",
        )

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=True)
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_wsl_uses_scalar_path_on_first_attempt(self, run_mock, _is_wsl_mock):
        run_mock.return_value = subprocess.CompletedProcess([], 0, b"", b"")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = mux_audio_with_sigfpe_fallback(
                ffmpeg_path="/usr/bin/ffmpeg",
                video_path="video.mp4",
                output_path=Path(temp_dir) / "output.mp4",
                sample_rate=32000,
                channels=2,
                audio_pass=["-c:a", "aac"],
                audio_data=b"pcm",
                env={},
            )

        self.assertFalse(result.used_scalar_fallback)
        self.assertEqual(run_mock.call_count, 1)
        args = run_mock.call_args.args[0]
        self.assertEqual(args[args.index("-cpuflags") + 1], "0")
        self.assertEqual(
            run_mock.call_args.kwargs["timeout"],
            DEFAULT_WSL_FINALIZE_TIMEOUT,
        )

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=True)
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_wsl_scalar_failure_does_not_retry(self, run_mock, _is_wsl_mock):
        run_mock.return_value = subprocess.CompletedProcess(
            [], -signal.SIGFPE, b"", b"boom"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            with self.assertRaisesRegex(AudioMuxError, "scalar path"):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=output_path,
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"pcm",
                    env={},
                )
            self.assertFalse(output_path.exists())

        self.assertEqual(run_mock.call_count, 1)

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=False)
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_non_sigfpe_failure_does_not_retry_and_removes_partial_output(
        self,
        run_mock,
        _is_wsl_mock,
    ):
        def fail(args, **kwargs):
            Path(args[-1]).write_bytes(b"partial")
            return subprocess.CompletedProcess(args, 1, b"", b"boom")

        run_mock.side_effect = fail
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            with self.assertRaisesRegex(AudioMuxError, "boom"):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=output_path,
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"pcm",
                    env={},
                )
            self.assertFalse(output_path.exists())

        self.assertEqual(run_mock.call_count, 1)

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=False)
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_failing_scalar_fallback_raises_and_removes_partial_output(
        self,
        run_mock,
        _is_wsl_mock,
    ):
        calls = 0

        def fail(args, **kwargs):
            nonlocal calls
            calls += 1
            Path(args[-1]).write_bytes(b"partial")
            return subprocess.CompletedProcess(
                args,
                -signal.SIGFPE if calls == 1 else 1,
                b"",
                b"primary" if calls == 1 else b"fallback",
            )

        run_mock.side_effect = fail
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            with self.assertRaisesRegex(
                AudioMuxError,
                "(?s)scalar fallback also failed.*fallback",
            ):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=output_path,
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"pcm",
                    env={},
                )
            self.assertFalse(output_path.exists())

        self.assertEqual(run_mock.call_count, 2)

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=False)
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_timeout_removes_partial_output(self, run_mock, _is_wsl_mock):
        def time_out(args, **kwargs):
            Path(args[-1]).write_bytes(b"partial")
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])

        run_mock.side_effect = time_out
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            with mock.patch.dict(
                os.environ,
                {"VHS_FFMPEG_FINALIZE_TIMEOUT": "5"},
                clear=False,
            ):
                with self.assertRaisesRegex(AudioMuxError, "timed out"):
                    mux_audio_with_sigfpe_fallback(
                        ffmpeg_path="/usr/bin/ffmpeg",
                        video_path="video.mp4",
                        output_path=output_path,
                        sample_rate=32000,
                        channels=2,
                        audio_pass=["-c:a", "aac"],
                        audio_data=b"pcm",
                        env={},
                    )
            self.assertFalse(output_path.exists())

    @mock.patch("videohelpersuite.audio_mux._is_wsl", return_value=False)
    @mock.patch("videohelpersuite.audio_mux.time.monotonic")
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_timeout_budget_is_shared_across_retry(
        self,
        run_mock,
        monotonic_mock,
        _is_wsl_mock,
    ):
        monotonic_mock.side_effect = [100.0, 101.0, 104.0]
        run_mock.side_effect = [
            subprocess.CompletedProcess([], -signal.SIGFPE, b"", b"primary"),
            subprocess.CompletedProcess([], 0, b"", b"fallback"),
        ]

        with mock.patch.dict(
            os.environ,
            {"VHS_FFMPEG_FINALIZE_TIMEOUT": "5"},
            clear=False,
        ):
            mux_audio_with_sigfpe_fallback(
                ffmpeg_path="/usr/bin/ffmpeg",
                video_path="video.mp4",
                output_path="output.mp4",
                sample_rate=32000,
                channels=2,
                audio_pass=["-c:a", "aac"],
                audio_data=b"pcm",
                env={},
            )

        self.assertEqual(run_mock.call_args_list[0].kwargs["timeout"], 4.0)
        self.assertEqual(run_mock.call_args_list[1].kwargs["timeout"], 1.0)


if __name__ == "__main__":
    unittest.main()
