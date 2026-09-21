"""Host side of the hand link: line protocol, heartbeat, and a mock for tests.

Wire format is newline-terminated ASCII so the link can be driven by hand from
any serial terminal while bringing the hardware up.

    host -> mcu                          mcu -> host
    G <primitive>   execute primitive    OK <echo>
    R               release (open)       ST <state> <primitive> <ms>
    S               hard stop, coast     ERR <reason>
    P               heartbeat            EV <what>            (asynchronous)
    ?               status request
    L <0..255>      global duty ceiling

The heartbeat is a safety interlock, not bookkeeping. These are tendon drives:
pull-only, with no position feedback in the base N20 build. If the host crashes
while a grip is closing, the motor stalls against the tendon and burns. Firmware
must coast every channel when heartbeats stop, so the host is required to keep
sending them -- see poll_and_beat().
"""
import time

from .decide import PRIMITIVES

BAUD = 115200
HEARTBEAT_SEC = 0.4        # must stay well under the firmware watchdog timeout
WATCHDOG_SEC = 1.5         # mirrored in firmware/esp32_hand/config.h


class MockHand:
    """In-process stand-in for the ESP32. Records traffic, answers like firmware."""

    def __init__(self, watchdog=WATCHDOG_SEC):
        self.sent = []
        self.rx = []
        self.watchdog = watchdog
        self.primitive = "open"
        self.state = "idle"
        self.released_by_watchdog = False
        self.last_rx = None
        self.clock = 0.0

    def write_line(self, line):
        self.sent.append((self.clock, line))
        self.last_rx = self.clock
        verb, _, arg = line.partition(" ")
        if verb == "G":
            if arg not in PRIMITIVES:
                self.rx.append(f"ERR unknown primitive {arg}")
                return
            self.primitive, self.state = arg, "moving"
            self.rx.append(f"OK G {arg}")
        elif verb == "R":
            self.primitive, self.state = "open", "moving"
            self.rx.append("OK R")
        elif verb == "S":
            self.state = "coast"
            self.rx.append("OK S")
        elif verb == "P":
            self.rx.append("OK P")
        elif verb == "?":
            self.rx.append(f"ST {self.state} {self.primitive} 0")
        elif verb == "L":
            self.rx.append(f"OK L {arg}")
        else:
            self.rx.append(f"ERR bad verb {verb}")

    def read_lines(self):
        out, self.rx = self.rx, []
        return out

    def advance(self, dt):
        """Move the mock clock; trips the watchdog exactly as firmware would."""
        self.clock += dt
        if self.last_rx is not None and (self.clock - self.last_rx) > self.watchdog:
            if self.state != "coast":
                self.state = "coast"
                self.primitive = "open"
                self.released_by_watchdog = True
                self.rx.append("EV watchdog_release")

    def close(self):
        pass


class SerialHand:
    """pyserial transport to the ESP32."""

    def __init__(self, port, baud=BAUD, timeout=0.05):
        import serial

        self.ser = serial.Serial(port, baud, timeout=timeout)
        self._buf = b""
        time.sleep(0.3)      # ESP32 resets on port open; ignore its boot chatter
        self.ser.reset_input_buffer()

    def write_line(self, line):
        self.ser.write((line + "\n").encode())

    def read_lines(self):
        self._buf += self.ser.read(4096)
        out = []
        while b"\n" in self._buf:
            raw, _, self._buf = self._buf.partition(b"\n")
            s = raw.decode(errors="replace").strip()
            if s:
                out.append(s)
        return out

    def close(self):
        try:
            self.write_line("S")
        finally:
            self.ser.close()


class Hand:
    """High-level hand controller over any transport exposing write/read_line(s)."""

    def __init__(self, link, heartbeat=HEARTBEAT_SEC, clock=time.monotonic):
        self.link = link
        self.heartbeat = heartbeat
        self.clock = clock
        self._last_beat = -1e9
        self.log = []
        self.primitive = "open"

    def _send(self, line):
        self.link.write_line(line)
        self._last_beat = self.clock()
        self.log.append((self._last_beat, line))

    def go(self, primitive):
        if primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {primitive!r}")
        self.primitive = primitive
        self._send(f"G {primitive}")

    def release(self):
        self.primitive = "open"
        self._send("R")

    def stop(self):
        self._send("S")

    def status(self):
        self._send("?")

    def set_duty_limit(self, duty):
        self._send(f"L {int(max(0, min(255, duty)))}")

    def poll_and_beat(self):
        """Call every loop iteration: sends a heartbeat when due, drains replies."""
        now = self.clock()
        if now - self._last_beat >= self.heartbeat:
            self._send("P")
        return self.link.read_lines()

    def apply(self, cmd):
        """Execute a decide.Command. CYCLE only highlights, so it sends nothing."""
        if cmd.kind == "confirm":
            self.go(cmd.primitive)
            return True
        if cmd.kind == "release":
            self.release()
            return True
        return False

    def close(self):
        self.link.close()
