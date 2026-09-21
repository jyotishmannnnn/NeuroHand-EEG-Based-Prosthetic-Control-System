"""Tier-1 BCI -> hand runner: blink/jaw gestures drive grasp primitives.

    # no cap, no hardware: synthetic gestures through a mock hand
    python run_tier1.py --source synth --mock --calib 5

    # replay a recorded session to check it produces no spurious commands
    python run_tier1.py --source "C:/.../LiveSession/09-57-47" --mock --calib 20

    # live: app in STREAMING-LSL mode, ESP32 on COM5
    python run_tier1.py --source lsl --port COM5

Controls, once calibrated: a short blink cycles the highlighted primitive, a held
blink executes it, a jaw clench opens the hand immediately.
"""
import argparse
import sys
import time

from neuro.decide import PRIMITIVES, DecideConfig, Decider
from neuro.detect import DetectConfig, Tier1Detector
from neuro.hand import Hand, MockHand, SerialHand
from neuro.source import open_source


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="synth",
                   help="'lsl', 'synth', or a LiveSession directory to replay")
    p.add_argument("--port", help="ESP32 serial port, e.g. COM5")
    p.add_argument("--mock", action="store_true", help="use in-process mock hand")
    p.add_argument("--calib", type=float, default=20.0,
                   help="seconds of rest used to learn the baseline")
    p.add_argument("--blink-z", type=float, default=12.0)
    p.add_argument("--jaw-z", type=float, default=5.0)
    p.add_argument("--no-ratio-gate", action="store_true",
                   help="disable the frontal/posterior artifact gate")
    p.add_argument("--force", action="store_true",
                   help="arm even if frontal contact fails the quality check")
    p.add_argument("--duration", type=float, default=60.0, help="synth length, seconds")
    p.add_argument("--realtime", action="store_true", help="pace replay to wall clock")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def synth_events(duration, calib):
    """Rest for calibration, then a scripted gesture sequence for a visible demo."""
    t0 = calib + 4.0
    return [
        (t0, "blink"),                 # cycle  open  -> power
        (t0 + 3.0, "blink"),           # cycle  power -> pinch
        (t0 + 6.0, "blink_long"),      # confirm: execute pinch
        (t0 + 10.0, "jaw"),            # release: open the hand
    ]


def quality_report(det, src):
    if det.quality is None:
        return "(no quality metrics)"
    m = det.quality
    rows = [f"{'ch':>4} {'rms':>10} {'50Hz/sig':>9} {'flat%':>7}"]
    for i, n in enumerate(src.names):
        rows.append(f"{n:>4} {m['rms'][i]:10.1f} {m['mains_ratio'][i]:9.2f} "
                    f"{m['flat_frac'][i] * 100:7.1f}")
    return "\n".join(rows)


def make_hand(args, stream_clock):
    """Build the hand link.

    The mock runs its watchdog on stream time, so in a faster-than-realtime replay
    its clock must also drive the heartbeat; using wall time there would make every
    replay look like a dead link and trip the watchdog spuriously.
    """
    if args.mock or not args.port:
        if not args.mock:
            print("no --port given; using mock hand", file=sys.stderr)
        clock = time.monotonic if args.realtime else stream_clock
        return Hand(MockHand(), clock=clock), True
    return Hand(SerialHand(args.port)), False


def main(argv=None):
    args = build_args(argv)

    kw = dict(realtime=args.realtime)
    if args.source == "synth":
        kw.update(duration=args.duration,
                  events=synth_events(args.duration, args.calib))
    src = open_source(args.source, **kw)

    det = Tier1Detector(src.names, src.fs, DetectConfig(
        blink_z=args.blink_z, jaw_z=args.jaw_z, calib_sec=args.calib,
        use_ratio_gate=not args.no_ratio_gate,
        require_quality=not args.force))
    dec = Decider(DecideConfig())
    now = [0.0]
    hand, is_mock = make_hand(args, lambda: now[0])

    say = (lambda *a: None) if args.quiet else print
    say(f"source={args.source} fs={src.fs:.1f} ch={src.names}")
    say(f"calibrating on {args.calib:.0f}s of rest -- hold still, no blinking")

    events, commands = [], []
    announced = False
    t_stream = 0.0
    try:
        for chunk, stamps in src:
            t_stream = now[0] = float(stamps[-1])
            evs = det.push(chunk)
            if not det.calibrated:
                hand.poll_and_beat()
                continue
            if not announced:
                announced = True
                say(f"\nbaseline locked at {t_stream:.1f}s")
                say(quality_report(det, src))
                if not det.armed:
                    say("\nNOT ARMED -- frontal contact failed the quality check:")
                    for r in det.quality_reasons:
                        say(f"  - {r}")
                    say("Fix contact and re-run, or pass --force to override "
                        "(the hand will then be driven by whatever this noise does).")
                    break
                if det.quality_reasons:
                    say(f"armed with warnings: {det.quality_reasons}")
                say(f"armed. highlighted: {dec.primitive}")
            for ev in evs:
                events.append(ev)
                say(f"  {ev}")
            for cmd in dec.feed(evs, now=t_stream):
                commands.append(cmd)
                acted = hand.apply(cmd)
                say(f"  {cmd}{'  [sent]' if acted else '  [highlight only]'}")
            for line in hand.poll_and_beat():
                if not line.startswith("OK P"):
                    say(f"  <- {line}")
            if is_mock:
                hand.link.advance(len(chunk) / src.fs)
    except KeyboardInterrupt:
        say("\ninterrupted")
    finally:
        hand.stop()
        hand.close()
        src.close()

    say(f"\n{t_stream:.1f}s processed | {len(events)} events | "
        f"{len(commands)} commands | {dec.dropped} dropped while busy")
    return events, commands


if __name__ == "__main__":
    main()
