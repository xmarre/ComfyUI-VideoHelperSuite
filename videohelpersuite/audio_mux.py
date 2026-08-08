import math
import os
import platform
import shlex
import signal
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path


ENCODE_ARGS = ("utf-8", "backslashreplace")
DEFAULT_WSL_FINALIZE_TIMEOUT = 120.0


@dataclass(frozen=True)
class AudioMuxResult:
    stderr: str
    used_scalar_fallback: bool
    used_codec_fallback: bool = False


class AudioMuxError(RuntimeError):
    pass


def _is_wsl():
    """Return True when running inside Windows Subsystem for Linux."""
    if os.name != "posix":
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    release = platform.release().lower()
    return "microsoft" in release or "wsl" in release


def configured_finalize_timeout():
    raw_timeout = os.environ.get("VHS_FFMPEG_FINALIZE_TIMEOUT")
    if raw_timeout is None:
        return DEFAULT_WSL_FINALIZE_TIMEOUT if _is_wsl() else None
    try:
        timeout = float(raw_timeout or 0)
    except ValueError:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        return None
    return timeout


def _validate_output_duration(value):
    if value is None:
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioMuxError("audio mux output duration must be positive") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AudioMuxError("audio mux output duration must be positive")
    return duration


def _replace_audio_codec(audio_pass, codec):
    args = list(audio_pass)
    for option in ("-c:a", "-codec:a", "-acodec"):
        try:
            index = args.index(option)
        except ValueError:
            continue
        if index + 1 >= len(args):
            raise AudioMuxError(f"{option} is missing its codec value")
        args[index + 1] = codec
        return args
    return ["-c:a", codec, *args]


def build_audio_mux_args(
    ffmpeg_path,
    video_path,
    output_path,
    sample_rate,
    channels,
    audio_pass,
    *,
    disable_cpu_flags=False,
    audio_input_path=None,
    output_duration=None,
):
    output_duration = _validate_output_duration(output_duration)
    args = [ffmpeg_path, "-nostdin", "-v", "error", "-y"]
    if disable_cpu_flags:
        args += ["-cpuflags", "0"]
    args += ["-i", str(video_path)]
    if audio_input_path is None:
        args += [
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-f",
            "s16le",
            "-i",
            "-",
        ]
    else:
        args += ["-i", str(audio_input_path)]
    args += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    args += list(audio_pass)
    args += ["-threads:a", "1"]
    if output_duration is None:
        args += ["-shortest"]
    else:
        args += ["-t", f"{output_duration:.9f}"]
    args += [str(output_path)]
    return args


def _run_mux(args, audio_data, env, timeout):
    try:
        completed = subprocess.run(
            args,
            input=audio_data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioMuxError(
            "ffmpeg timed out while muxing audio: " + shlex.join(args)
        ) from exc
    return completed.returncode, completed.stderr.decode(*ENCODE_ARGS)


def _remaining_timeout(deadline):
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AudioMuxError("ffmpeg audio mux exhausted its configured timeout")
    return remaining


def _write_pcm_wav(path, audio_data, sample_rate, channels):
    if channels < 1:
        raise AudioMuxError("audio channel count must be positive")
    frame_size = channels * 2
    if len(audio_data) % frame_size != 0:
        raise AudioMuxError(
            "PCM16 audio byte length is not aligned to the channel frame size"
        )

    try:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
    except (OSError, wave.Error) as exc:
        raise AudioMuxError(
            f"failed to create seekable WAV fallback input at {path}"
        ) from exc


def _run_seekable_scalar_mux(
    ffmpeg_path,
    video_path,
    output_path,
    sample_rate,
    channels,
    audio_pass,
    audio_data,
    env,
    deadline,
    output_duration=None,
):
    try:
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=".vhs-audio-",
                dir=output_path.parent,
            )
        )
    except OSError as exc:
        raise AudioMuxError(
            "failed to create the seekable WAV fallback directory near "
            f"{output_path}"
        ) from exc

    try:
        wav_path = temp_dir / "audio.wav"
        _write_pcm_wav(wav_path, audio_data, sample_rate, channels)
        args = build_audio_mux_args(
            ffmpeg_path,
            video_path,
            output_path,
            sample_rate,
            channels,
            audio_pass,
            disable_cpu_flags=True,
            audio_input_path=wav_path,
            output_duration=output_duration,
        )
        returncode, stderr = _run_mux(
            args,
            None,
            env,
            _remaining_timeout(deadline),
        )
        return returncode, stderr, args
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def mux_audio_with_sigfpe_fallback(
    ffmpeg_path,
    video_path,
    output_path,
    sample_rate,
    channels,
    audio_pass,
    audio_data,
    env,
    output_duration=None,
):
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    output_duration = _validate_output_duration(output_duration)

    timeout = configured_finalize_timeout()
    deadline = time.monotonic() + timeout if timeout is not None else None

    # WSL has now produced SIGFPE with raw PCM, seekable WAV, normal CPU flags,
    # and -cpuflags 0. Keep the safer seekable/scalar topology, but when the
    # video duration is known avoid FFmpeg's -shortest scheduler entirely.
    if _is_wsl():
        try:
            returncode, stderr, args = _run_seekable_scalar_mux(
                ffmpeg_path,
                video_path,
                output_path,
                sample_rate,
                channels,
                audio_pass,
                audio_data,
                env,
                deadline,
                output_duration=output_duration,
            )
        except AudioMuxError:
            output_path.unlink(missing_ok=True)
            raise
        if returncode == 0:
            return AudioMuxResult(
                stderr=stderr,
                used_scalar_fallback=False,
                used_codec_fallback=False,
            )

        if returncode != -signal.SIGFPE:
            output_path.unlink(missing_ok=True)
            raise AudioMuxError(
                f"ffmpeg exited with status {returncode} while muxing audio "
                "on the WSL seekable scalar path.\n"
                f"Command: {shlex.join(args)}\n{stderr}"
            )

        # The observed WSL failure survives a seekable WAV and -cpuflags 0.
        # Remove the native AAC encoder from the second attempt as well. ALAC
        # is valid in MP4 and gives us a genuinely independent codec path while
        # preserving the already encoded video stream with -c:v copy.
        output_path.unlink(missing_ok=True)
        alac_pass = _replace_audio_codec(audio_pass, "alac")
        try:
            fallback_returncode, fallback_stderr, fallback_args = (
                _run_seekable_scalar_mux(
                    ffmpeg_path,
                    video_path,
                    output_path,
                    sample_rate,
                    channels,
                    alac_pass,
                    audio_data,
                    env,
                    deadline,
                    output_duration=output_duration,
                )
            )
        except AudioMuxError:
            output_path.unlink(missing_ok=True)
            raise

        if fallback_returncode != 0:
            output_path.unlink(missing_ok=True)
            raise AudioMuxError(
                "ffmpeg received SIGFPE on the WSL seekable scalar mux and "
                "the ALAC codec fallback also failed with status "
                f"{fallback_returncode}.\n"
                f"Primary command: {shlex.join(args)}\n"
                f"Primary stderr:\n{stderr}\n"
                f"ALAC fallback command: {shlex.join(fallback_args)}\n"
                f"ALAC fallback stderr:\n{fallback_stderr}"
            )

        return AudioMuxResult(
            stderr=fallback_stderr,
            used_scalar_fallback=True,
            used_codec_fallback=True,
        )

    primary_args = build_audio_mux_args(
        ffmpeg_path,
        video_path,
        output_path,
        sample_rate,
        channels,
        audio_pass,
    )
    try:
        returncode, stderr = _run_mux(
            primary_args,
            audio_data,
            env,
            _remaining_timeout(deadline),
        )
    except AudioMuxError:
        output_path.unlink(missing_ok=True)
        raise
    if returncode == 0:
        return AudioMuxResult(stderr=stderr, used_scalar_fallback=False)

    if returncode != -signal.SIGFPE:
        output_path.unlink(missing_ok=True)
        raise AudioMuxError(
            f"ffmpeg exited with status {returncode} while muxing audio.\n"
            f"Command: {shlex.join(primary_args)}\n{stderr}"
        )

    output_path.unlink(missing_ok=True)
    try:
        fallback_returncode, fallback_stderr, fallback_args = (
            _run_seekable_scalar_mux(
                ffmpeg_path,
                video_path,
                output_path,
                sample_rate,
                channels,
                audio_pass,
                audio_data,
                env,
                deadline,
            )
        )
    except AudioMuxError:
        output_path.unlink(missing_ok=True)
        raise

    if fallback_returncode != 0:
        output_path.unlink(missing_ok=True)
        raise AudioMuxError(
            "ffmpeg received SIGFPE during the normal audio mux and the "
            "seekable WAV fallback also failed with status "
            f"{fallback_returncode}.\n"
            f"Primary command: {shlex.join(primary_args)}\n"
            f"Primary stderr:\n{stderr}\n"
            f"Fallback command: {shlex.join(fallback_args)}\n"
            f"Fallback stderr:\n{fallback_stderr}"
        )

    return AudioMuxResult(
        stderr=fallback_stderr,
        used_scalar_fallback=True,
    )
