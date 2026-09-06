"""Cross-platform process metrics.

`resource` is POSIX-only, so importing it at module scope breaks the whole
package on Windows with ModuleNotFoundError. It is also inconsistent where it
does exist: `ru_maxrss` is kilobytes on Linux but bytes on macOS, so the obvious
`/1024` is wrong on one of them.

This module isolates all of that. Peak RSS is a reporting nicety for the cost
table -- it must never be able to stop a run -- so every path degrades to NaN
rather than raising.

Resolution order:
  1. psutil, if installed (most accurate, all platforms)
  2. Windows: GetProcessMemoryInfo via ctypes, no dependency
  3. POSIX: resource.getrusage, with the platform-correct unit
  4. NaN
"""

from __future__ import annotations

import platform
import sys

_IS_WINDOWS = sys.platform.startswith("win")
_IS_MACOS = sys.platform == "darwin"

NAN = float("nan")


def _peak_rss_psutil() -> float:
    try:
        import psutil                                   # optional
    except Exception:
        return NAN
    try:
        info = psutil.Process().memory_info()
        # peak_wset on Windows, rss elsewhere (psutil has no portable peak)
        val = getattr(info, "peak_wset", None)
        if val is None:
            val = info.rss
        return float(val) / (1024.0 * 1024.0)
    except Exception:
        return NAN


def _peak_rss_windows() -> float:
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return NAN
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        return NAN


def _peak_rss_posix() -> float:
    try:
        import resource                                  # POSIX only
    except Exception:
        return NAN
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes; macOS reports bytes.
        return float(raw) / (1024.0 * 1024.0) if _IS_MACOS else float(raw) / 1024.0
    except Exception:
        return NAN


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB. NaN if unavailable."""
    val = _peak_rss_psutil()
    if val == val:
        return val
    val = _peak_rss_windows() if _IS_WINDOWS else _peak_rss_posix()
    return val


def describe_host() -> dict:
    """Machine facts for the cost table. Never raises."""
    import os
    try:
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "python": platform.python_version(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
            "gpu_used": False,
        }
    except Exception:
        return {"platform": "unknown", "gpu_used": False}
