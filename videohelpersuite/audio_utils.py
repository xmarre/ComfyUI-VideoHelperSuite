import math
from dataclasses import dataclass

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
    """Return a CPU float32 waveform that is safe to stream to ffmpeg.

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
