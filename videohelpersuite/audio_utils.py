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
    samples_per_channel: int


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


def sanitize_audio_waveform(waveform):
    """Return a CPU float32 waveform that is safe to serialize.

    ComfyUI AUDIO waveforms use the shape ``[batch, channels, samples]``.
    Video Combine only supports a single audio item, so reject ambiguous
    batched inputs instead of silently flattening them.
    """
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
    """Convert one sanitized ComfyUI waveform to interleaved signed PCM16.

    Padding is performed in Python so the ffmpeg mux pass does not need the
    floating-point ``apad`` filter. The returned byte order is explicitly
    little-endian for the ffmpeg ``s16le`` demuxer.
    """
    if isinstance(minimum_samples, bool):
        raise ValueError("minimum_samples must be a non-negative integer")
    try:
        minimum_samples_int = int(minimum_samples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("minimum_samples must be a non-negative integer") from exc
    if minimum_samples_int != minimum_samples or minimum_samples_int < 0:
        raise ValueError("minimum_samples must be a non-negative integer")

    sanitized = sanitize_audio_waveform(waveform).waveform
    current_samples = sanitized.size(2)
    if current_samples < minimum_samples_int:
        sanitized = torch.nn.functional.pad(
            sanitized,
            (0, minimum_samples_int - current_samples),
        )

    scaled = torch.where(
        sanitized < 0,
        sanitized * 32768.0,
        sanitized * 32767.0,
    )
    pcm = scaled.round().clamp(-32768, 32767).to(torch.int16)
    interleaved = pcm.squeeze(0).transpose(0, 1).contiguous().numpy()
    data = interleaved.astype(np.dtype("<i2"), copy=False).tobytes()

    return PcmAudioBuffer(
        data=data,
        channels=int(sanitized.size(1)),
        samples_per_channel=int(sanitized.size(2)),
    )
