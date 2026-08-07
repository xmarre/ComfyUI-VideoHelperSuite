import os
import signal
import subprocess
import tempfile
import unittest
import wave
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

    def test_builds_seekable_audio_input_without_raw_pcm_options(self):
        args = build_audio_mux_args(
            "/usr/bin/ffmpeg",
            "video.mp4",
            "output.mp4",
            32000,
            2,
            ["-c:a", "aac"],
            disable_cpu_flags=True,
            audio_input_path="audio.wav",
        )

        self.assertNotIn("s16le", args)
        self.assertNotIn("-ar", args)
        self.assertNotIn("-ac", args)
        self.assertEqual(args[args.index("-cpuflags") + 1], "0")
        input_positions = [index for index, arg in enumerate(args) if arg == "-i"]
        self.assertEqual(args[input_positions[-1] + 1], "audio.wav")


class AudioMuxFallbackTests(unittest.TestCase):
    def setUp(self):
        self._wsl_patcher = mock.patch(
            "videohelpersuite.audio_mux._is_wsl",
            return_value=False,
        )
        self._is_wsl_mock = self._wsl_patcher.start()

    def tearDown(self):
        self._wsl_patcher.stop()

    def test_wsl_has_bounded_default_timeout(self):
        self._is_wsl_mock.return_value = True
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                configured_finalize_timeout(),
                DEFAULT_WSL_FINALIZE_TIMEOUT,
            )

    def test_explicit_zero_disables_wsl_default_timeout(self):
        self._is_wsl_mock.return_value = True
        with mock.patch.dict(
            os.environ,
            {"VHS_FFMPEG_FINALIZE_TIMEOUT": "0"},
            clear=True,
        ):
            self.assertIsNone(configured_finalize_timeout())

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_wsl_uses_seekable_scalar_path_on_first_attempt(self, run_mock):
        self._is_wsl_mock.return_value = True
        first_wav = None

        def succeed(args, **kwargs):
            nonlocal first_wav
            input_positions = [
                index for index, arg in enumerate(args) if arg == "-i"
            ]
            first_wav = Path(args[input_positions[-1] + 1])
            self.assertTrue(first_wav.is_file())
            self.assertIsNone(kwargs["input"])
            with wave.open(str(first_wav), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 32000)
                self.assertEqual(wav_file.readframes(2), b"\0" * 8)
            return subprocess.CompletedProcess(args, 0, b"", b"")

        run_mock.side_effect = succeed
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {}, clear=True):
                result = mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=Path(temp_dir) / "output.mp4",
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"\0" * 8,
                    env={},
                )

        self.assertFalse(result.used_scalar_fallback)
        self.assertEqual(run_mock.call_count, 1)
        args = run_mock.call_args.args[0]
        self.assertEqual(args[args.index("-cpuflags") + 1], "0")
        self.assertNotIn("s16le", args)
        self.assertNotIn("-ar", args)
        self.assertNotIn("-ac", args)
        timeout = run_mock.call_args.kwargs["timeout"]
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, DEFAULT_WSL_FINALIZE_TIMEOUT)
        self.assertIsNotNone(first_wav)
        self.assertFalse(first_wav.exists())

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_wsl_seekable_scalar_failure_does_not_retry(self, run_mock):
        self._is_wsl_mock.return_value = True
        run_mock.return_value = subprocess.CompletedProcess([], 1, b"", b"boom")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            with self.assertRaisesRegex(AudioMuxError, "WSL seekable scalar path"):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=output_path,
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"\0" * 8,
                    env={},
                )
            self.assertFalse(output_path.exists())

        self.assertEqual(run_mock.call_count, 1)

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_sigfpe_retries_with_seekable_wav(self, run_mock):
        fallback_wav = None

        def side_effect(args, **kwargs):
            nonlocal fallback_wav
            if run_mock.call_count == 1:
                return subprocess.CompletedProcess(
                    args,
                    -signal.SIGFPE,
                    b"",
                    b"primary",
                )

            input_positions = [
                index for index, arg in enumerate(args) if arg == "-i"
            ]
            fallback_wav = Path(args[input_positions[-1] + 1])
            self.assertTrue(fallback_wav.is_file())
            self.assertIsNone(kwargs["input"])
            with wave.open(str(fallback_wav), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 32000)
                self.assertEqual(wav_file.readframes(2), b"\0" * 8)
            return subprocess.CompletedProcess(args, 0, b"", b"fallback")

        run_mock.side_effect = side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            result = mux_audio_with_sigfpe_fallback(
                ffmpeg_path="/usr/bin/ffmpeg",
                video_path="video.mp4",
                output_path=Path(temp_dir) / "output.mp4",
                sample_rate=32000,
                channels=2,
                audio_pass=["-c:a", "aac"],
                audio_data=b"\0" * 8,
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
        self.assertIsNotNone(fallback_wav)
        self.assertFalse(fallback_wav.exists())

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_non_sigfpe_failure_does_not_retry_and_removes_partial_output(
        self,
        run_mock,
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

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_failing_seekable_fallback_raises_and_removes_partial_output(
        self,
        run_mock,
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
                "(?s)seekable WAV fallback also failed.*fallback",
            ):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=output_path,
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"\0" * 8,
                    env={},
                )
            self.assertFalse(output_path.exists())

        self.assertEqual(run_mock.call_count, 2)

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_rejects_unaligned_pcm_before_seekable_fallback(self, run_mock):
        run_mock.return_value = subprocess.CompletedProcess(
            [],
            -signal.SIGFPE,
            b"",
            b"primary",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(AudioMuxError, "not aligned"):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=Path(temp_dir) / "output.mp4",
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"pcm",
                    env={},
                )

        self.assertEqual(run_mock.call_count, 1)

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_temp_directory_failure_is_reported_as_audio_mux_error(
        self,
        run_mock,
    ):
        run_mock.return_value = subprocess.CompletedProcess(
            [],
            -signal.SIGFPE,
            b"",
            b"primary",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "videohelpersuite.audio_mux.tempfile.mkdtemp",
                side_effect=OSError("disk error"),
            ):
                with self.assertRaisesRegex(AudioMuxError, "fallback directory"):
                    mux_audio_with_sigfpe_fallback(
                        ffmpeg_path="/usr/bin/ffmpeg",
                        video_path="video.mp4",
                        output_path=Path(temp_dir) / "output.mp4",
                        sample_rate=32000,
                        channels=2,
                        audio_pass=["-c:a", "aac"],
                        audio_data=b"\0" * 8,
                        env={},
                    )

        self.assertEqual(run_mock.call_count, 1)

    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_timeout_removes_partial_output(self, run_mock):
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

    @mock.patch("videohelpersuite.audio_mux.time.monotonic")
    @mock.patch("videohelpersuite.audio_mux.subprocess.run")
    def test_timeout_budget_is_shared_across_retry(self, run_mock, monotonic_mock):
        monotonic_mock.side_effect = [100.0, 101.0, 104.0]
        run_mock.side_effect = [
            subprocess.CompletedProcess([], -signal.SIGFPE, b"", b"primary"),
            subprocess.CompletedProcess([], 0, b"", b"fallback"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ,
                {"VHS_FFMPEG_FINALIZE_TIMEOUT": "5"},
                clear=False,
            ):
                mux_audio_with_sigfpe_fallback(
                    ffmpeg_path="/usr/bin/ffmpeg",
                    video_path="video.mp4",
                    output_path=Path(temp_dir) / "output.mp4",
                    sample_rate=32000,
                    channels=2,
                    audio_pass=["-c:a", "aac"],
                    audio_data=b"\0" * 8,
                    env={},
                )

        self.assertEqual(run_mock.call_args_list[0].kwargs["timeout"], 4.0)
        self.assertEqual(run_mock.call_args_list[1].kwargs["timeout"], 1.0)


if __name__ == "__main__":
    unittest.main()
