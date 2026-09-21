"""Discover NeuroAnalytics LSL streams and print live sample rates.

Run NeuroAnalytics in STREAMING-LSL mode first. Streams are named
<UserName>_UnFilter / _EEG / _PPG / _IMU / _Marker.

Usage: python lsl_probe.py [seconds]
"""
import sys
import time

from pylsl import StreamInlet, resolve_streams

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

print("resolving LSL streams (5 s)...")
streams = resolve_streams(wait_time=5.0)
if not streams:
    print("none found. Is NeuroAnalytics in STREAMING-LSL mode and streaming?")
    sys.exit(1)

inlets = []
for s in streams:
    info = f"{s.name():<28} type={s.type():<8} ch={s.channel_count():<3} nominal={s.nominal_srate():>6.1f} Hz  fmt={s.channel_format()}  host={s.hostname()}"
    print(info)
    inlets.append((s.name(), StreamInlet(s, max_buflen=60)))

# Pull channel labels from stream metadata if present
for name, inlet in inlets:
    desc = inlet.info().desc()
    ch = desc.child("channels").child("channel")
    labels = []
    while ch.name() == "channel":
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    if labels:
        print(f"{name} labels: {labels}")

print(f"\ncounting samples for {secs:.0f} s...")
counts = {n: 0 for n, _ in inlets}
first = {}
t0 = time.time()
while time.time() - t0 < secs:
    for name, inlet in inlets:
        chunk, ts = inlet.pull_chunk(timeout=0.0)
        if chunk:
            counts[name] += len(chunk)
            if name not in first:
                first[name] = chunk[0]
    time.sleep(0.01)

print()
for name, _ in inlets:
    print(f"{name:<28} {counts[name] / secs:7.1f} samples/s   first sample: {first.get(name)}")
