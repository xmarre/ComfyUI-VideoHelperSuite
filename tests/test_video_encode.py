import signal
import unittest

from videohelpersuite.video_encode import (
    FFmpegProcessError,
    detect_video_encoder,
    scalarize_software_encode_args,
    supports_scalar_software_fallback,
)


class VideoEncodeRecoveryTests(unittest.TestCase):
    def test_detects_software_encoder(self):
        args = ["ffmpeg", "-i", "-", "-c:v", "libx264", "-crf", "20"]
        self.assertEqual(detect_video_encoder(args), "libx264")
        self.assertTrue(supports_scalar_software_fallback(args))

    def test_rejects_non_software_fallback_encoder(self):
        args = ["ffmpeg", "-i", "-", "-c:v", "h264_nvenc"]
        self.assertFalse(supports_scalar_software_fallback(args))
        with self.assertRaisesRegex(ValueError, "no scalar software fallback"):
            scalarize_software_encode_args(args)

    def test_scalarizes_x264_without_changing_encode_semantics(self):
        args = [
            "ffmpeg",
            "-nostdin",
            "-f",
            "rawvideo",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p10le",
            "-crf",
            "19",
        ]
        recovered = scalarize_software_encode_args(args)

        self.assertEqual(detect_video_encoder(recovered), "libx264")
        self.assertEqual(recovered[recovered.index("-pix_fmt") + 1], "yuv420p10le")
        self.assertEqual(recovered[recovered.index("-crf") + 1], "19")
        self.assertEqual(recovered[recovered.index("-cpuflags") + 1], "0")
        self.assertEqual(recovered[recovered.index("-filter_threads") + 1], "1")
        self.assertLess(recovered.index("-cpuflags"), recovered.index("-i"))
        self.assertLess(recovered.index("-filter_threads"), recovered.index("-i"))
        self.assertEqual(recovered[recovered.index("-threads:v") + 1], "1")
        self.assertEqual(
            recovered[recovered.index("-x264-params") + 1],
            "asm=0:threads=1",
        )

    def test_scalarizes_x265_and_preserves_existing_params(self):
        args = [
            "ffmpeg",
            "-i",
            "-",
            "-c:v",
            "libx265",
            "-threads",
            "4",
            "-x265-params",
            "log-level=quiet:frame-threads=4:asm=avx2",
            "-crf",
            "22",
        ]
        recovered = scalarize_software_encode_args(args)

        self.assertEqual(recovered[recovered.index("-threads") + 1], "1")
        self.assertEqual(
            recovered[recovered.index("-x265-params") + 1],
            "log-level=quiet:asm=0:frame-threads=1",
        )
        self.assertEqual(recovered[recovered.index("-crf") + 1], "22")

    def test_scalarization_is_idempotent(self):
        args = ["ffmpeg", "-i", "-", "-c:v", "libx264"]
        once = scalarize_software_encode_args(args)
        twice = scalarize_software_encode_args(once)
        self.assertEqual(once, twice)

    def test_process_error_keeps_signal_command_and_stderr(self):
        command = ["/opt/ffmpeg-vhs", "-i", "-", "out.mp4"]
        error = FFmpegProcessError(
            context="saving video",
            returncode=-signal.SIGFPE,
            command=command,
            stderr="native crash detail\n",
            file_path="out.mp4",
        )

        self.assertTrue(error.is_sigfpe)
        self.assertEqual(error.signal_number, signal.SIGFPE)
        self.assertEqual(error.file_path, "out.mp4")
        self.assertIn("status -8 (SIGFPE)", str(error))
        self.assertIn("/opt/ffmpeg-vhs -i - out.mp4", str(error))
        self.assertIn("native crash detail", str(error))


if __name__ == "__main__":
    unittest.main()
