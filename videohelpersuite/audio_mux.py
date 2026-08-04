import math
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


ENCODE_ARGS = ("utf-8", "backslashreplace")


@dataclass(frozen=True)
class AudioMuxResult:
    stderr: str
    used_scalar_fallback: bool


class AudioMuxError(RuntimeError):
    pass


def _finalize_timeout():
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
):
    args = [ffmpeg_path, "-nostdin", "-v", "error", "-y"]
    if disable_cpu_flags:
        args += ["-cpuflags", "0"]
    args += [
        "-i",
        str(video_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-f",
        "s16le",
        "-i",
        "-",
        "-c:v",
        "copy",
    ]
    args += list(audio_pass)
    args += ["-threads:a", "1", "-shortest", str(output_path)]
    return args


def _run_mux(args, audio_data, env):
    try:
        completed = subprocess.run(
            args,
            input=audio_data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            timeout=_finalize_timeout(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioMuxError(
            "ffmpeg timed out while muxing audio: " + shlex.join(args)
        ) from exc
    return completed.returncode, completed.stderr.decode(*ENCODE_ARGS)


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

    primary_args = build_audio_mux_args(
        ffmpeg_path,
        video_path,
        output_path,
        sample_rate,
        channels,
        audio_pass,
    )
    returncode, stderr = _run_mux(primary_args, audio_data, env)
    if returncode == 0:
        return AudioMuxResult(stderr=stderr, used_scalar_fallback=False)

    if returncode != -signal.SIGFPE:
        raise AudioMuxError(
            f"ffmpeg exited with status {returncode} while muxing audio.\n"
            f"Command: {shlex.join(primary_args)}\n{stderr}"
        )

    output_path.unlink(missing_ok=True)
    fallback_args = build_audio_mux_args(
        ffmpeg_path,
        video_path,
        output_path,
        sample_rate,
        channels,
        audio_pass,
        disable_cpu_flags=True,
    )
    fallback_returncode, fallback_stderr = _run_mux(
        fallback_args,
        audio_data,
        env,
    )
    if fallback_returncode != 0:
        raise AudioMuxError(
            "ffmpeg received SIGFPE during the normal audio mux and the "
            f"scalar fallback also failed with status {fallback_returncode}.\n"
            f"Primary command: {shlex.join(primary_args)}\n"
            f"Primary stderr:\n{stderr}\n"
            f"Fallback command: {shlex.join(fallback_args)}\n"
            f"Fallback stderr:\n{fallback_stderr}"
        )

    return AudioMuxResult(
        stderr=fallback_stderr,
        used_scalar_fallback=True,
    )
