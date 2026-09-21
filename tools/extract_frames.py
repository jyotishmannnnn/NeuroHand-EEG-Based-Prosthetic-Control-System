"""Extract evenly spaced frames from a screen recording for inspection."""
import sys
from pathlib import Path

import cv2

src = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
out = Path(sys.argv[3]) if len(sys.argv) > 3 else src.parent / "frames"
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(src))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{src.name}: {w}x{h} @ {fps:.2f} fps, {total} frames, {total / fps:.1f} s")

for i in range(n):
    idx = int((i + 0.5) * total / n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        print(f"frame {idx}: read failed")
        continue
    t = idx / fps
    path = out / f"{src.stem}_t{t:06.1f}s.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {path.name}")
cap.release()
