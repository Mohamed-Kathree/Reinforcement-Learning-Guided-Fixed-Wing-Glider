"""
link.py -- the Pi's end of the ESP32 UART link (Pi header pin 8 TXD -> ESP32
GPIO 33, pin 10 RXD <- ESP32 GPIO 32, 115200 8N1).

Reads PiTelemetry with the generated StreamParser (sync + CRC8, never blocks on
a partial frame), tracks sequence gaps (reported back to the ESP32 in every
PiSetpoint), and writes PiSetpoint frames.

`port` is anything with read(n) -> bytes and write(bytes): a pyserial Serial
on the Pi, or an in-memory stand-in in the tests and the simulator check.
"""

from __future__ import annotations

from onboard.vendor import packets as P


class EspLink:
    def __init__(self, port) -> None:
        self.port = port
        self.parser = P.StreamParser()
        self.latest = None              # newest PiTelemetry
        self.rx_frames = 0
        self.tlm_gaps = 0
        self._last_seq = None
        self._tx_seq = 0

    def poll(self) -> list:
        """Read everything available; returns the new PiTelemetry frames, oldest first."""
        data = self.port.read(4096)
        if not data:
            return []
        out = []
        for pkt in self.parser.feed(data):
            if not isinstance(pkt, P.PiTelemetry):
                continue
            if self._last_seq is not None:
                step = (pkt.seq - self._last_seq) & 0xFFFF
                if 1 < step < 0x8000:
                    self.tlm_gaps += step - 1
            self._last_seq = pkt.seq
            self.rx_frames += 1
            self.latest = pkt
            out.append(pkt)
        return out

    def send(self, roll_sp: float, speed_cmd: float, controller: int) -> bytes:
        self._tx_seq = (self._tx_seq + 1) & 0xFFFF
        frame = P.PiSetpoint(seq=self._tx_seq, controller=controller, roll_sp=roll_sp,
                             speed_cmd=speed_cmd, tlm_gaps=min(self.tlm_gaps, 0xFFFF)).encode()
        self.port.write(frame)
        return frame

    @property
    def dropped(self) -> int:
        return self.parser.dropped


def open_serial(device: str, baud: int = 115200):
    import serial                       # pyserial, installed on the Pi
    return serial.Serial(device, baud, timeout=0, write_timeout=0)
