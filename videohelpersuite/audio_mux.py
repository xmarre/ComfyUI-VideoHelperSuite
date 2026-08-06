import math
import os
import shlex
import signal
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path


ENCODE_ARGS = ("utf-8", "backslashreplace")


@dataclass(frozen=True)
class AudioMuxResult:
    stderr: str
    used_scalar_fallback: bool


class AudioMuxError(RuntimeError):
    pass


def configured_finalize_timeout():
    try:
        timeout = float(os.environ.get("VHS_FFMPEG_FINALIZE_TIMEOUT", "0") or 0)
    except ValueError:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        return None
    return timeout


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
):
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
    args += ["-threads:a", "1", "-shortest", str(output_path)]
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


def mux_audio_with_sigfpe_fallback(
    ffmpeg_path,
    video_path,
    output_path,
    sample_rate,
    channels,
    audio_pass,
    audio_data,
    env,
):
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)

    timeout = configured_finalize_timeout()
    deadline = time.monotonic() + timeout if timeout is not None else None

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
        with tempfile.TemporaryDirectory(
            prefix=".vhs-audio-",
            dir=output_path.parent,
        ) as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            _write_pcm_wav(wav_path, audio_data, sample_rate, channels)
            fallback_args = build_audio_mux_args(
                ffmpeg_path,
                video_path,
                output_path,
                sample_rate,
                channels,
                audio_pass,
                disable_cpu_flags=True,
                audio_input_path=wav_path,
            )
            fallback_returncode, fallback_stderr = _run_mux(
                fallback_args,
                None,
                env,
                _remaining_timeout(deadline),
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
