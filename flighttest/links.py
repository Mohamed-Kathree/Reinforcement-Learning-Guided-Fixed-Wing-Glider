"""
links.py
========
Byte transports between the ground app and the ground ESP32.

  StubLink    in-process StubGlider; used whenever no hardware is attached.
  SerialLink  the ground ESP32 on USB serial. HARDWARE SEAM, UNTESTED: written
              against pyserial's documented API but never run against the
              receiver. It assumes the receiver firmware is a transparent bridge:
              every ESP-NOW frame it receives is written to USB unchanged, and
              every byte written to USB is sent over ESP-NOW unchanged. The
              ground app does all framing (sync hunt, CRC) itself, so a bridge
              that adds no headers is the simplest thing to build. If the bridge
              adds RSSI, it should append it in a separate frame type; RSSI is
              reported as unavailable until then.

Every link is non-blocking: read() returns whatever bytes are available now.
"""

from __future__ import annotations

from flighttest.stub_glider import StubGlider


class Link:
    kind = "none"
    port = ""

    def read(self) -> bytes:
        raise NotImplementedError

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def step(self, now_ms: int) -> None:
        """Advance any simulation; no-op for real hardware."""

    def close(self) -> None:
        pass

    @property
    def rssi_dbm(self) -> float | None:
        return None


class StubLink(Link):
    kind = "stub"
    port = "stub (simulated glider)"

    def __init__(self, glider: StubGlider) -> None:
        self.glider = glider
        self._now = 0

    def read(self) -> bytes:
        return self.glider.read()

    def write(self, data: bytes) -> None:
        self.glider.receive(data, self._now)

    def step(self, now_ms: int) -> None:
        self._now = now_ms
        self.glider.step(now_ms)


class SerialLink(Link):
    """HARDWARE SEAM: untested. See module docstring for the bridge assumption."""

    kind = "serial"

    def __init__(self, port: str, baud: int = 57600) -> None:   # firmware/ground_station GS_BAUD
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise RuntimeError("pyserial is not installed: pip install pyserial") from exc
        self.port = port
        self._ser = serial.Serial(port, baud, timeout=0, write_timeout=0.05)

    def read(self) -> bytes:
        n = self._ser.in_waiting
        return self._ser.read(n) if n else b""

    def write(self, data: bytes) -> None:
        self._ser.write(data)

    def close(self) -> None:
        self._ser.close()


def list_serial_ports() -> dict:
    """Serial ports visible to the OS. Never raises: reports why if it cannot list."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return {"available": False, "reason": "pyserial is not installed (pip install pyserial)", "ports": []}
    return {"available": True, "reason": None,
            "ports": [{"device": p.device, "description": p.description, "hwid": p.hwid}
                      for p in list_ports.comports()]}
