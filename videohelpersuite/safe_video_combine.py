import math
import os
import shutil
import subprocess
from pathlib import Path

from .audio_mux import AudioMuxError, mux_audio_with_sigfpe_fallback
from .audio_utils import (
    sanitize_audio_waveform,
    validate_sample_rate,
    waveform_to_pcm_s16le,
)
from .logger import logger
from .nodes import VideoCombine, apply_format_widgets
from .utils import ffmpeg_path


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


def _ffprobe_path():
    if ffmpeg_path:
        sibling = Path(ffmpeg_path).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def _probe_video_duration(video_path):
    probe = _ffprobe_path()
    if probe is None:
        return None
    completed = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _format_options(format_name, manual_format_widgets, kwargs):
    format_kwargs = dict(kwargs)
    if manual_format_widgets is not None:
        format_kwargs.update(manual_format_widgets)
    return apply_format_widgets(format_name, format_kwargs)


class SafeVideoCombine(VideoCombine):
    """Video Combine variant with a hardened, recoverable audio mux path."""

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
        if audio is None or not format.startswith("video/"):
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

        audio = _sanitize_audio_input(audio)
        format_name = format.split("/", 1)[1]
        video_format = _format_options(format_name, manual_format_widgets, kwargs)

        # Let the established VideoCombine implementation create the video,
        # but deliberately bypass its raw-f32/apad audio subprocess. The
        # hardened mux below uses deterministic PCM16 and can retry SIGFPE.
        result = super().combine_video(
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
            audio=None,
            unique_id=unique_id,
            manual_format_widgets=manual_format_widgets,
            meta_batch=meta_batch,
            vae=vae,
            **kwargs,
        )

        if "gifski_pass" in video_format:
            return result
        if not isinstance(result, dict) or "result" not in result:
            return result
        if result.get("ui", {}).get("unfinished_batch"):
            return result

        save_flag, existing_outputs = result["result"][0]
        output_files = list(existing_outputs)
        if not output_files:
            return result

        video_path = Path(output_files[-1])
        if not video_path.is_file():
            raise RuntimeError(
                "Video Combine completed without an accessible video file: "
                f"{video_path}"
            )

        output_path = video_path.with_name(
            f"{video_path.stem}-audio{video_path.suffix}"
        )
        sample_rate = audio["sample_rate"]
        trim_to_audio = video_format.get("trim_to_audio", "False") != "False"

        minimum_samples = 0
        if not trim_to_audio:
            duration = _probe_video_duration(video_path)
            if duration is None:
                logger.warn(
                    "Could not probe the video duration; audio will not be "
                    "pre-padded before muxing"
                )
            else:
                # Match the previous one-second safety margin, but create the
                # silence in Python instead of using ffmpeg's apad filter.
                minimum_samples = math.ceil((duration + 1.0) * sample_rate)

        pcm = waveform_to_pcm_s16le(
            audio["waveform"],
            minimum_samples=minimum_samples,
        )
        audio_pass = video_format.get("audio_pass", ["-c:a", "libopus"])
        env = os.environ.copy()
        if "environment" in video_format:
            env.update(video_format["environment"])

        try:
            mux_result = mux_audio_with_sigfpe_fallback(
                ffmpeg_path=ffmpeg_path,
                video_path=video_path,
                output_path=output_path,
                sample_rate=sample_rate,
                channels=pcm.channels,
                audio_pass=audio_pass,
                audio_data=pcm.data,
                env=env,
            )
        except AudioMuxError as exc:
            raise Exception(
                f"{exc}\nVideo-only output was preserved at: {video_path}"
            ) from exc

        if mux_result.used_scalar_fallback:
            logger.warn(
                "ffmpeg audio mux received SIGFPE; recovered automatically "
                "with CPU SIMD disabled"
            )
        if mux_result.stderr:
            print(mux_result.stderr, end="")

        output_files.append(str(output_path))
        result["result"] = ((save_flag, output_files),)

        previews = result.get("ui", {}).get("gifs", [])
        if previews:
            previews[0]["filename"] = output_path.name
            previews[0]["fullpath"] = str(output_path)

        extra_options = (
            extra_pnginfo.get("workflow", {}).get("extra", {})
            if extra_pnginfo is not None
            else {}
        )
        if extra_options.get("VHS_KeepIntermediate", True) is False:
            video_path.unlink(missing_ok=True)

        return result
