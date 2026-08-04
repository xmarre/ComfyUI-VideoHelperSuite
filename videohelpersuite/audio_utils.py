import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AudioSanitizationResult:
    waveform: torch.Tensor
    nonfinite_samples: int
    clipped_samples: int
    finite_peak: float

    @property
    def changed(self):
        return self.nonfinite_samples > 0 or self.clipped_samples > 0


@dataclass(frozen=True)
class PcmAudioBuffer:
    data: bytes
    channels: int
    samples: int
    nonfinite_samples: int
    clipped_samples: int
    finite_peak: float

    @property
    def changed(self):
        return self.nonfinite_samples > 0 or self.clipped_samples > 0


def validate_sample_rate(value):
    """Return a positive integral sample rate without lossy conversion."""
    if isinstance(value, bool):
        raise ValueError("audio sample rate must be a positive integer")

    try:
        numeric_value = float(value)
        sample_rate = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("audio sample rate must be a positive integer") from exc

    if (
        not math.isfinite(numeric_value)
        or numeric_value != sample_rate
        or sample_rate <= 0
    ):
        raise ValueError("audio sample rate must be a positive integer")

    return sample_rate


def _validate_minimum_samples(value):
    if isinstance(value, bool):
        raise ValueError("minimum_samples must be a non-negative integer")
    try:
        numeric_value = float(value)
        minimum_samples = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("minimum_samples must be a non-negative integer") from exc
    if (
        not math.isfinite(numeric_value)
        or numeric_value != minimum_samples
        or minimum_samples < 0
    ):
        raise ValueError("minimum_samples must be a non-negative integer")
    return minimum_samples


def sanitize_audio_waveform(waveform):
    """Return a CPU float32 waveform that is safe to serialize."""
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("audio waveform must be a torch.Tensor")
    if waveform.ndim != 3 or waveform.size(0) != 1:
        raise ValueError(
            "audio waveform must have shape [1, channels, samples]; "
            f"received {tuple(waveform.shape)}"
        )
    if waveform.size(1) < 1 or waveform.size(2) < 1:
        raise ValueError("audio waveform must contain at least one channel and sample")

    sanitized = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    finite_mask = torch.isfinite(sanitized)
    nonfinite_samples = int((~finite_mask).sum().item())

    if finite_mask.any():
        finite_peak = float(sanitized[finite_mask].abs().max().item())
    else:
        finite_peak = 0.0

    clipped_samples = int((finite_mask & (sanitized.abs() > 1.0)).sum().item())

    if nonfinite_samples:
        sanitized = torch.nan_to_num(
            sanitized,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
    if clipped_samples:
        sanitized = sanitized.clamp(-1.0, 1.0)

    return AudioSanitizationResult(
        waveform=sanitized.contiguous(),
        nonfinite_samples=nonfinite_samples,
        clipped_samples=clipped_samples,
        finite_peak=finite_peak,
    )


def waveform_to_pcm_s16le(waveform, minimum_samples=0):
    """Sanitize, optionally pad, interleave, and encode a waveform as PCM16LE."""
    minimum_samples = _validate_minimum_samples(minimum_samples)
    sanitized = sanitize_audio_waveform(waveform)
    pcm_waveform = sanitized.waveform

    channels = int(pcm_waveform.size(1))
    samples = int(pcm_waveform.size(2))
    if samples < minimum_samples:
        padding = torch.zeros(
            (1, channels, minimum_samples - samples),
            dtype=pcm_waveform.dtype,
            device=pcm_waveform.device,
        )
        pcm_waveform = torch.cat((pcm_waveform, padding), dim=2)
        samples = minimum_samples

    interleaved = (
        pcm_waveform.squeeze(0)
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )
    scaled = np.where(
        interleaved < 0,
        np.rint(interleaved * 32768.0),
        np.rint(interleaved * 32767.0),
    )
    pcm = np.clip(scaled, -32768, 32767).astype("<i2", copy=False)

    return PcmAudioBuffer(
        data=pcm.tobytes(),
        channels=channels,
        samples=samples,
        nonfinite_samples=sanitized.nonfinite_samples,
        clipped_samples=sanitized.clipped_samples,
        finite_peak=sanitized.finite_peak,
    )
