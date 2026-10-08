"""Small, dependency-free styling for human-readable training messages."""

from __future__ import annotations

import builtins
import os
import re
import sys
from typing import TextIO


def supports_color(stream: TextIO) -> bool:
    return (
        not os.environ.get("NO_COLOR")
        and os.environ.get("TERM") != "dumb"
        and bool(getattr(stream, "isatty", lambda: False)())
    )


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if text else text


def _style_line(line: str) -> str:
    # Do not nest styles around messages already formatted by another logger.
    if not line.strip() or "\033[" in line:
        return line
    stripped = line.lstrip()
    if re.search(r"\b(error|failed|failure)\b", stripped, re.IGNORECASE):
        return _paint(line, "31")
    if stripped.lower().startswith(("warning", "clearing existing")):
        return _paint(line, "33")
    if stripped.startswith(("Starting ", "Loss terms:", "Optimization completed")):
        return _paint(line, "1;36")
    if stripped.startswith(("Saved ", "Copied ", "Loaded ", "Started ", "Outputs saved",
                            "Initial parameters written", "Final parameters written")):
        label, separator, value = line.partition(":")
        return _paint(label, "32") + separator + value
    if stripped.startswith("[") and "]" in line:
        tag, remainder = line.split("]", 1)
        return _paint(tag + "]", "35") + remainder
    if ":" in line:
        label, separator, value = line.partition(":")
        value_style = "1" if re.fullmatch(r"\s*[+-]?[\d.]+(?:[eE][+-]?\d+)?\s*", value) else ""
        if "total loss" in label.lower():
            return _paint(label + separator + value, "1;32")
        return _paint(label, "36") + separator + (_paint(value, value_style) if value_style else value)
    return line


def print_status(*values: object, sep: str | None = " ", end: str | None = "\n",
                 file: TextIO | None = None, flush: bool = False) -> None:
    """Print with terminal-only accents; preserve the original text in log files."""
    stream = sys.stdout if file is None else file
    if not supports_color(stream):
        builtins.print(*values, sep=sep, end=end, file=stream, flush=flush)
        return
    text = (" " if sep is None else sep).join(str(value) for value in values)
    builtins.print("\n".join(_style_line(line) for line in text.split("\n")),
                   end=end, file=stream, flush=flush)
