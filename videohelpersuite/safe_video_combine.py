import math
import os
import shutil
import subprocess
from pathlib import Path

from .audio_mux import (
    AudioMuxError,
    _is_wsl,
    configured_finalize_timeout,
    mux_audio_with_sigfpe_fallback,
)
from .audio_utils import validate_sample_rate, waveform_to_pcm_s16le
from .logger import logger
from .nodes import VideoCombine, apply_format_widgets
from .utils import ffmpeg_path
from .video_encode import FFmpegProcessError, supports_scalar_software_fallback


DEFAULT_FFPROBE_TIMEOUT = 30.0


def _validate_audio_input(audio):
    if not isinstance(audio, dict):
        raise TypeError("audio input must be a dictionary")
    if "waveform" not in audio:
        raise ValueError("audio input is missing the waveform")
    if "sample_rate" not in audio:
        raise ValueError("audio input is missing the sample rate")

    validated_audio = dict(audio)
    validated_audio["sample_rate"] = validate_sample_rate(audio["sample_rate"])
    return validated_audio


def _ffprobe_path():
    if ffmpeg_path:
        ffmpeg = Path(ffmpeg_path)
        sibling = ffmpeg.with_name(f"ffprobe{ffmpeg.suffix}")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def _probe_video_duration(video_path):
    probe = _ffprobe_path()
    if probe is None:
        return None

    timeout = configured_finalize_timeout() or DEFAULT_FFPROBE_TIMEOUT
    try:
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
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
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


def _cleanup_failed_encode_attempt(error):
    if not error.file_path:
        return
    video_path = Path(error.file_path)
    video_path.unlink(missing_ok=True)
    video_path.with_suffix(".png").unlink(missing_ok=True)


def _raise_recovery_failure(primary_error, recovery_error):
    raise RuntimeError(
        f"{recovery_error}\n\n"
        "Primary ffmpeg failure before the recovery attempt:\n"
        f"{primary_error}"
    ) from recovery_error


class SafeVideoCombine(VideoCombine):
    """Video Combine variant with recoverable video encoding and audio muxing."""

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
        if not format.startswith("video/"):
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

        if audio is not None:
            audio = _validate_audio_input(audio)

        format_name = format.split("/", 1)[1]
        video_format = _format_options(format_name, manual_format_widgets, kwargs)
        base_video_kwargs = dict(kwargs)

        def run_video_only(extra_kwargs=None):
            attempt_kwargs = dict(base_video_kwargs)
            if extra_kwargs:
                attempt_kwargs.update(extra_kwargs)

            attempt_manual_widgets = manual_format_widgets
            if (
                manual_format_widgets is not None
                and extra_kwargs
                and "save_metadata" in extra_kwargs
            ):
                attempt_manual_widgets = dict(manual_format_widgets)
                attempt_manual_widgets["save_metadata"] = extra_kwargs["save_metadata"]

            return VideoCombine.combine_video(
                self,
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
                manual_format_widgets=attempt_manual_widgets,
                meta_batch=meta_batch,
                vae=vae,
                **attempt_kwargs,
            )

        attempt_options = {}
        primary_error = None
        while True:
            try:
                result = run_video_only(attempt_options)
                break
            except FFmpegProcessError as error:
                if primary_error is None:
                    primary_error = error

                # Meta-batches retain an ffmpeg generator across executions.
                # Replaying one failed batch would corrupt that retained state.
                if meta_batch is not None:
                    raise

                scalar_retry = (
                    error.is_sigfpe
                    and _is_wsl()
                    and not attempt_options.get("_vhs_scalar_software_encode", False)
                    and supports_scalar_software_fallback(error.command)
                )
                if scalar_retry:
                    _cleanup_failed_encode_attempt(error)
                    attempt_options["_vhs_scalar_software_encode"] = True
                    logger.warn(
                        "ffmpeg software video encoding received SIGFPE on WSL; "
                        "retrying the same codec, pixel format, bit depth, and "
                        "quality settings with FFmpeg and encoder SIMD disabled "
                        "and a single encoder thread"
                    )
                    continue

                metadata_retry = (
                    error.metadata_attempt
                    and attempt_options.get("save_metadata") is not False
                    and (
                        not error.is_sigfpe
                        or attempt_options.get("_vhs_scalar_software_encode", False)
                    )
                )
                if metadata_retry:
                    _cleanup_failed_encode_attempt(error)
                    attempt_options["save_metadata"] = False
                    logger.warn(
                        "ffmpeg rejected the metadata-bearing video encode; "
                        "retrying the complete frame sequence without embedded "
                        "video metadata"
                    )
                    continue

                if primary_error is error:
                    raise
                _raise_recovery_failure(primary_error, error)

        if audio is None:
            return result

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

        # Probe once for both Python-side padding and an explicit mux duration.
        # Supplying -t lets the WSL path avoid FFmpeg's -shortest scheduler while
        # preserving the same final-duration semantics when probing succeeds.
        video_duration = _probe_video_duration(video_path)
        if video_duration is None:
            logger.warn(
                "Could not probe the video duration; audio will not be "
                "pre-padded and the mux must fall back to -shortest"
            )

        minimum_samples = 0
        if not trim_to_audio and video_duration is not None:
            # Match the previous one-second safety margin, but create the
            # silence in Python instead of using ffmpeg's apad filter.
            minimum_samples = math.ceil((video_duration + 1.0) * sample_rate)

        pcm = waveform_to_pcm_s16le(
            audio["waveform"],
            minimum_samples=minimum_samples,
        )
        if pcm.changed:
            logger.warn(
                "Sanitized audio before ffmpeg mux: "
                f"replaced {pcm.nonfinite_samples} non-finite sample(s), "
                f"clipped {pcm.clipped_samples} out-of-range sample(s), "
                f"finite peak={pcm.finite_peak:.6g}"
            )

        output_duration = None
        if video_duration is not None:
            if trim_to_audio:
                audio_duration = pcm.samples / sample_rate
                output_duration = min(video_duration, audio_duration)
            else:
                output_duration = video_duration

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
                output_duration=output_duration,
            )
        except AudioMuxError as exc:
            raise Exception(
                f"{exc}\nVideo-only output was preserved at: {video_path}"
            ) from exc

        if mux_result.used_codec_fallback:
            logger.warn(
                "ffmpeg AAC mux received SIGFPE on WSL; recovered with the "
                "seekable scalar ALAC-in-MP4 fallback"
            )
        elif mux_result.used_scalar_fallback:
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
