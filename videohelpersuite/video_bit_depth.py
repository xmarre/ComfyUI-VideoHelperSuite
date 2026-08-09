"""Bit-depth handling shared by Video Combine output formats."""


SUPPORTED_VIDEO_BIT_DEPTHS = (8, 10)


def normalize_video_bit_depth(bit_depth):
    """Return a supported integer bit depth from a widget/manual value."""
    if isinstance(bit_depth, bool):
        raise ValueError("bit_depth must be 8 or 10")

    if isinstance(bit_depth, str):
        value = bit_depth.strip().lower().removesuffix("-bit").removesuffix("bit")
    else:
        value = bit_depth

    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bit_depth must be 8 or 10") from exc

    if normalized not in SUPPORTED_VIDEO_BIT_DEPTHS:
        raise ValueError("bit_depth must be 8 or 10")
    return normalized


def apply_video_bit_depth(video_format, bit_depth):
    """Apply a format's bit-depth pixel format and matching RGB pipe depth.

    Formats opt in with a ``bit_depths`` mapping. Ten-bit YUV must receive
    16-bit RGB input; sending rgb24 would irreversibly quantize the image to
    eight bits before FFmpeg performs the RGB-to-YUV conversion.
    """
    bit_depths = video_format.get("bit_depths")
    if bit_depths is None:
        return video_format

    normalized = normalize_video_bit_depth(bit_depth)
    pixel_format = bit_depths.get(str(normalized))
    if pixel_format is None:
        supported = ", ".join(sorted(bit_depths))
        raise ValueError(
            f"Selected video format does not support {normalized}-bit output "
            f"(supported: {supported})"
        )

    main_pass = video_format.get("main_pass")
    if not isinstance(main_pass, list):
        raise ValueError("Bit-depth-aware video format is missing main_pass")
    try:
        pixel_format_index = main_pass.index("-pix_fmt") + 1
    except ValueError as exc:
        raise ValueError(
            "Bit-depth-aware video format is missing a -pix_fmt argument"
        ) from exc
    if pixel_format_index >= len(main_pass):
        raise ValueError(
            "Bit-depth-aware video format has no value after -pix_fmt"
        )

    main_pass[pixel_format_index] = pixel_format
    video_format["input_color_depth"] = "16bit" if normalized == 10 else "8bit"
    return video_format
