"""Builds firmware/protocol/*.c plus tests/c/rlg_host_helpers.c into a host shared
library and loads it with ctypes. Returns None when no C compiler is available
(tests then skip, and say so)."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = [os.path.join(ROOT, "firmware", "protocol", "rlg_crc8.c"),
       os.path.join(ROOT, "firmware", "protocol", "rlg_safety.c"),
       os.path.join(ROOT, "tests", "c", "rlg_host_helpers.c")]
_LIB = None


def compiler() -> str | None:
    return shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")


def load():
    global _LIB
    if _LIB is not None:
        return _LIB
    cc = compiler()
    if cc is None:
        return None
    ext = ".dll" if sys.platform == "win32" else ".so"
    out = os.path.join(tempfile.mkdtemp(prefix="rlg_host_"), "rlg" + ext)
    cmd = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-shared", "-fPIC",
           "-I", os.path.join(ROOT, "firmware", "protocol"), *SRC, "-o", out]
    if sys.platform == "win32":
        cmd.remove("-fPIC")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(out)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.rlg_crc8.argtypes = [u8p, ctypes.c_size_t]
    lib.rlg_crc8.restype = ctypes.c_uint8
    lib.rlg_test_telemetry_size.restype = ctypes.c_size_t
    lib.rlg_test_command_size.restype = ctypes.c_size_t
    lib.rlg_test_decode_telemetry.argtypes = [u8p, ctypes.POINTER(ctypes.c_double)]
    lib.rlg_test_decode_command.argtypes = [u8p, ctypes.POINTER(ctypes.c_uint32)]
    lib.rlg_safety_sizeof.restype = ctypes.c_size_t
    lib.rlg_safety_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32]
    lib.rlg_safety_on_command.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_bool, u8p, ctypes.c_size_t]
    lib.rlg_safety_on_command.restype = ctypes.c_int
    lib.rlg_safety_tick.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.rlg_safety_flight_output.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
    lib.rlg_safety_flight_output.restype = ctypes.c_bool
    lib.rlg_safety_status_bits.argtypes = [ctypes.c_void_p]
    lib.rlg_safety_status_bits.restype = ctypes.c_uint32
    lib.rlg_safety_mode.argtypes = [ctypes.c_void_p]
    lib.rlg_safety_mode.restype = ctypes.c_uint8
    lib.rlg_safety_armed.argtypes = [ctypes.c_void_p]
    lib.rlg_safety_armed.restype = ctypes.c_bool
    lib.rlg_safety_servo.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rlg_safety_servo.restype = ctypes.c_uint16
    _LIB = lib
    return lib


def buf(data: bytes):
    return (ctypes.c_uint8 * max(len(data), 1)).from_buffer_copy(data if data else b"\x00")
