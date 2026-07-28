from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


class GStreamerUnavailableError(RuntimeError):
    """Raised when the optional GStreamer Python runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class GStreamerRuntime:
    Gst: Any
    GLib: Any


_REQUIRED_ELEMENTS = (
    "appsrc",
    "gdppay",
    "tcpserversink",
    "tcpclientsrc",
    "gdpdepay",
    "appsink",
)


@lru_cache(maxsize=1)
def require_gstreamer() -> GStreamerRuntime:
    """Load GStreamer lazily and verify the packet-transport elements."""

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)
        missing = tuple(
            name for name in _REQUIRED_ELEMENTS if Gst.ElementFactory.find(name) is None
        )
    except Exception as exc:
        raise GStreamerUnavailableError(
            "GStreamer packet transport requires PyGObject, loadable GLib libraries, and the Gst 1.0 typelib"
        ) from exc

    if missing:
        names = ", ".join(missing)
        raise GStreamerUnavailableError(
            f"GStreamer is installed but required packet-transport elements are missing: {names}"
        )
    return GStreamerRuntime(Gst=Gst, GLib=GLib)


def gstreamer_available() -> bool:
    """Return whether the GDP-over-TCP transport can be constructed."""

    try:
        require_gstreamer()
    except GStreamerUnavailableError:
        return False
    return True
