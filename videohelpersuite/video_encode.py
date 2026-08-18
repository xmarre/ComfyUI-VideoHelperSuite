import shlex
import signal


SOFTWARE_ENCODERS_WITH_SCALAR_FALLBACK = {"libx264", "libx265"}


class FFmpegProcessError(RuntimeError):
    """Structured ffmpeg encode failure with the information needed for recovery."""

    def __init__(
        self,
        *,
        context,
        returncode,
        command,
        stderr="",
        file_path=None,
        metadata_attempt=False,
    ):
        self.context = context
        self.returncode = returncode
        self.command = tuple(command)
        self.stderr = stderr
        self.file_path = file_path
        self.metadata_attempt = metadata_attempt
        super().__init__(
            format_ffmpeg_process_error(
                context=context,
                returncode=returncode,
                command=command,
                stderr=stderr,
            )
        )

    @property
    def signal_number(self):
        if isinstance(self.returncode, int) and self.returncode < 0:
            return -self.returncode
        return None

    @property
    def is_sigfpe(self):
        return self.returncode == -signal.SIGFPE


def _exit_status_description(returncode):
    if returncode is None:
        return "unknown status"
    if returncode >= 0:
        return f"status {returncode}"
    signum = -returncode
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = f"signal {signum}"
    return f"status {returncode} ({signal_name})"


def format_ffmpeg_process_error(*, context, returncode, command, stderr=""):
    """Format an ffmpeg failure without losing a signal-only exit or command line."""
    message = (
        f"ffmpeg exited with {_exit_status_description(returncode)} while {context}.\n"
        f"Command: {shlex.join([str(value) for value in command])}"
    )
    if stderr:
        message += f"\n{stderr.rstrip()}"
    return message


def detect_video_encoder(args):
    """Return the selected video encoder from an ffmpeg argument vector."""
    for index, value in enumerate(args[:-1]):
        if value in {"-c:v", "-codec:v", "-vcodec"}:
            return str(args[index + 1])
    return None


def supports_scalar_software_fallback(args):
    return detect_video_encoder(args) in SOFTWARE_ENCODERS_WITH_SCALAR_FALLBACK


def _set_or_append_option(args, option_names, append_name, value):
    for index, option in enumerate(args[:-1]):
        if option in option_names:
            args[index + 1] = str(value)
            return
    args.extend([append_name, str(value)])


def _merge_codec_params(args, option_name, replacements):
    option_index = None
    for index, option in enumerate(args[:-1]):
        if option == option_name:
            option_index = index
            break

    existing = []
    if option_index is not None:
        existing = [part for part in str(args[option_index + 1]).split(":") if part]

    replacement_keys = set(replacements)
    preserved = []
    for part in existing:
        key = part.split("=", 1)[0]
        if key not in replacement_keys:
            preserved.append(part)
    preserved.extend(f"{key}={value}" for key, value in replacements.items())
    merged = ":".join(preserved)

    if option_index is None:
        args.extend([option_name, merged])
    else:
        args[option_index + 1] = merged


def scalarize_software_encode_args(args):
    """Build a conservative same-codec retry for WSL SIGFPE software encodes."""
    scalar_args = list(args)
    encoder = detect_video_encoder(scalar_args)
    if encoder not in SOFTWARE_ENCODERS_WITH_SCALAR_FALLBACK:
        raise ValueError(
            f"no scalar software fallback is defined for video encoder {encoder!r}"
        )

    _set_or_append_option(scalar_args, {"-cpuflags"}, "-cpuflags", "0")
    _set_or_append_option(
        scalar_args,
        {"-filter_threads"},
        "-filter_threads",
        "1",
    )
    _set_or_append_option(
        scalar_args,
        {"-threads", "-threads:v", "-threads:0"},
        "-threads:v",
        "1",
    )

    if encoder == "libx264":
        _merge_codec_params(
            scalar_args,
            "-x264-params",
            {"asm": "0", "threads": "1"},
        )
    elif encoder == "libx265":
        _merge_codec_params(
            scalar_args,
            "-x265-params",
            {"asm": "0", "frame-threads": "1"},
        )

    return scalar_args
