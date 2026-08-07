import math
import os
import platform
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ENCODE_ARGS = ("utf-8", "backslashreplace")
DEFAULT_WSL_FINALIZE_TIMEOUT = 120.0


@dataclass(frozen=True)
class AudioMuxResult:
    stderr: str
    used_scalar_fallback: bool


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
    force_scalar = _is_wsl()

    primary_args = build_audio_mux_args(
        ffmpeg_path,
        video_path,
        output_path,
        sample_rate,
        channels,
        audio_pass,
        disable_cpu_flags=force_scalar,
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

    # WSL has already used the scalar path. Retrying the same failed command
    # cannot recover anything and only extends the period in which a broken
    # ffmpeg process can hold resources.
    if force_scalar or returncode != -signal.SIGFPE:
        output_path.unlink(missing_ok=True)
        mode = " scalar" if force_scalar else ""
        raise AudioMuxError(
            f"ffmpeg exited with status {returncode} while muxing audio on the{mode} path.\n"
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
    try:
        fallback_returncode, fallback_stderr = _run_mux(
            fallback_args,
            audio_data,
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
