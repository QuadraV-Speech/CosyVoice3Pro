import os
import re


_PUNCTUATION_PARTS = re.compile(
    r".+?[。！？!?；;，,：:.]+|.+$", re.DOTALL)


def positive_env(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是正整数") from exc
    if value < 1:
        raise RuntimeError(f"{name} 必须是正整数")
    return value


def split_text(text, max_chars):
    text = text.strip()
    if not text:
        return []

    segments = []
    buffer = ""

    def append_hard_split(value):
        for index in range(0, len(value), max_chars):
            segment = value[index:index + max_chars].rstrip("，,").strip()
            if segment:
                segments.append(segment)

    for part in _PUNCTUATION_PARTS.findall(text):
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buffer:
                segments.append(buffer.rstrip("，,"))
                buffer = ""
            append_hard_split(part)
        elif len(buffer) + len(part) <= max_chars:
            buffer += part
        else:
            if buffer:
                segments.append(buffer.rstrip("，,"))
            buffer = part

    if buffer:
        segments.append(buffer.rstrip("，,"))
    return segments
