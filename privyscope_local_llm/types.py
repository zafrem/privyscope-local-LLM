"""Span type: use privyscope's own when installed so stages interoperate exactly."""
try:
    from privyscope._core.bioes import Span
except ImportError:  # standalone use (tests, no core installed)
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Span:  # type: ignore[no-redef]
        label: str
        start: int
        end: int
