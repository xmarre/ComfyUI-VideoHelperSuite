import unittest

import numpy as np
import torch

from videohelpersuite.audio_utils import (
    sanitize_audio_waveform,
    validate_sample_rate,
    waveform_to_pcm_s16le,
)


class ValidateSampleRateTests(unittest.TestCase):
    def test_accepts_positive_integral_sample_rate(self):
        self.assertEqual(validate_sample_rate(32000), 32000)

    def test_rejects_fractional_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_sample_rate(48000.5)

    def test_rejects_infinite_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_sample_rate(float("inf"))


class SanitizeAudioWaveformTests(unittest.TestCase):
    def test_preserves_valid_audio_and_normalizes_transport_requirements(self):
        source = torch.tensor(
            [[[0.25, -0.5, 1.0], [-1.0, 0.0, 0.75]]],
            dtype=torch.float64,
            requires_grad=True,
        )

        result = sanitize_audio_waveform(source)

        self.assertFalse(result.changed)
        self.assertEqual(result.nonfinite_samples, 0)
        self.assertEqual(result.clipped_samples, 0)
        self.assertEqual(result.finite_peak, 1.0)
        self.assertEqual(result.waveform.device.type, "cpu")
        self.assertEqual(result.waveform.dtype, torch.float32)
        self.assertTrue(result.waveform.is_contiguous())
        self.assertFalse(result.waveform.requires_grad)
        torch.testing.assert_close(result.waveform, source.detach().float())

    def test_replaces_nonfinite_samples_and_clips_out_of_range_samples(self):
        source = torch.tensor(
            [[[float("nan"), float("inf"), float("-inf"), 1.25, -1.5, 0.25]]]
        )

        result = sanitize_audio_waveform(source)

        self.assertTrue(result.changed)
        self.assertEqual(result.nonfinite_samples, 3)
        self.assertEqual(result.clipped_samples, 2)
        self.assertEqual(result.finite_peak, 1.5)
        torch.testing.assert_close(
            result.waveform,
            torch.tensor([[[0.0, 1.0, -1.0, 1.0, -1.0, 0.25]]]),
        )

    def test_rejects_multi_item_audio_batches(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            sanitize_audio_waveform(torch.zeros((2, 2, 32)))

    def test_rejects_empty_audio(self):
        with self.assertRaisesRegex(ValueError, "at least one channel and sample"):
            sanitize_audio_waveform(torch.zeros((1, 2, 0)))


class WaveformToPcmTests(unittest.TestCase):
    def test_interleaves_channels_and_quantizes_pcm16le(self):
        waveform = torch.tensor([[[0.0, 1.0], [-1.0, 0.5]]])

        result = waveform_to_pcm_s16le(waveform)
        decoded = np.frombuffer(result.data, dtype="<i2")

        self.assertEqual(result.channels, 2)
        self.assertEqual(result.samples, 2)
        self.assertEqual(decoded.tolist(), [0, -32768, 32767, 16384])

    def test_pads_with_silence(self):
        result = waveform_to_pcm_s16le(
            torch.tensor([[[0.5], [-0.5]]]),
            minimum_samples=3,
        )
        decoded = np.frombuffer(result.data, dtype="<i2")

        self.assertEqual(result.samples, 3)
        self.assertEqual(decoded.tolist(), [16384, -16384, 0, 0, 0, 0])

    def test_reports_sanitization_without_a_second_pass_contract(self):
        result = waveform_to_pcm_s16le(
            torch.tensor([[[float("nan"), 2.0]]])
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.nonfinite_samples, 1)
        self.assertEqual(result.clipped_samples, 1)
        self.assertEqual(result.finite_peak, 2.0)

    def test_rejects_fractional_minimum_samples(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            waveform_to_pcm_s16le(torch.zeros((1, 1, 1)), 1.5)

    def test_rejects_negative_minimum_samples(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            waveform_to_pcm_s16le(torch.zeros((1, 1, 1)), -1)

    def test_rejects_boolean_minimum_samples(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            waveform_to_pcm_s16le(torch.zeros((1, 1, 1)), True)


if __name__ == "__main__":
    unittest.main()
