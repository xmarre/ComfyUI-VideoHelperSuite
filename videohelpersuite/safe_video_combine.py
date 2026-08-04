from .audio_utils import sanitize_audio_waveform, validate_sample_rate
from .logger import logger
from .nodes import VideoCombine


def _sanitize_audio_input(audio):
    if not isinstance(audio, dict):
        raise TypeError("audio input must be a dictionary")
    if "waveform" not in audio:
        raise ValueError("audio input is missing the waveform")
    if "sample_rate" not in audio:
        raise ValueError("audio input is missing the sample rate")

    sample_rate = validate_sample_rate(audio["sample_rate"])
    result = sanitize_audio_waveform(audio["waveform"])
    if result.changed:
        logger.warn(
            "Sanitized audio before ffmpeg mux: "
            f"replaced {result.nonfinite_samples} non-finite sample(s), "
            f"clipped {result.clipped_samples} out-of-range sample(s), "
            f"finite peak={result.finite_peak:.6g}"
        )

    sanitized_audio = dict(audio)
    sanitized_audio["waveform"] = result.waveform
    sanitized_audio["sample_rate"] = sample_rate
    return sanitized_audio


class SafeVideoCombine(VideoCombine):
    """Video Combine variant that validates audio before ffmpeg sees it."""

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        format="image/gif",  # noqa: A002
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None,
        **kwargs,
    ):
        if audio is not None:
            audio = _sanitize_audio_input(audio)

        return super().combine_video(
            frame_rate=frame_rate,
            loop_count=loop_count,
            images=images,
            latents=latents,
            filename_prefix=filename_prefix,
            format=format,
            pingpong=pingpong,
            save_output=save_output,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            audio=audio,
            unique_id=unique_id,
            manual_format_widgets=manual_format_widgets,
            meta_batch=meta_batch,
            vae=vae,
            **kwargs,
        )
