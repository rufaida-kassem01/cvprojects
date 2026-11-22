"""Select clips from the UCSD/WSDOT traffic database and convert them for StreamSight.

The original clips are 320x240 AVI files at 10 FPS, about 5 seconds each. The
database README says the first frame of every clip is corrupted, so it is
dropped. Clips are upscaled (default 2x) so the overlays and line preview are
easy to read, and written as MP4.

Usage (from the repo root, after unzipping the Kaggle archive to data/ucsd_raw):
    python tools/prepare_clips.py --raw data/ucsd_raw --out data/ucsd --per-class 3

Writes data/ucsd/<class>_<clipname>.mp4 and data/ucsd/manifest.csv. Fill in the
gt_dir_1 / gt_dir_2 columns with your manual counts for each clip.
"""
import argparse
import csv
import os
import random
from collections import defaultdict

import cv2


def read_info(raw):
    """info.txt columns: filename, date, timestamp, direction, day/night, weather, start frame,
    number of frames, class, notes (tab separated, '#' lines are comments)."""
    clips = []
    with open(os.path.join(raw, "info.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            clips.append({"name": c[0], "weather": c[5], "label": c[8]})
    return clips


def pick(clips, per_class, seed):
    """Pick per_class clips for each traffic class, spreading them over the weather types."""
    rng = random.Random(seed)
    chosen = []
    for label in ("light", "medium", "heavy"):
        by_weather = defaultdict(list)
        for c in clips:
            if c["label"] == label:
                by_weather[c["weather"]].append(c)
        for lst in by_weather.values():
            rng.shuffle(lst)
        weathers = sorted(by_weather, key=lambda w: -len(by_weather[w]))
        taken = 0
        while taken < per_class and any(by_weather[w] for w in weathers):
            for w in weathers:
                if taken < per_class and by_weather[w]:
                    chosen.append(by_weather[w].pop())
                    taken += 1
    return chosen


def convert(src, dst, scale):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ok, _ = cap.read()  # drop the corrupted first frame
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC))
        n += 1
    cap.release()
    writer.release()
    return n, fps, w, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="folder with the unzipped archive (info.txt and video/)")
    p.add_argument("--out", default="data/ucsd")
    p.add_argument("--per-class", type=int, default=3, help="clips to pick for each of light/medium/heavy")
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0, help="change to get a different selection")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chosen = pick(read_info(args.raw), args.per_class, args.seed)

    rows = []
    for c in chosen:
        src = os.path.join(args.raw, "video", c["name"] + ".avi")
        out_name = f"{c['label']}_{c['name']}.mp4"
        res = convert(src, os.path.join(args.out, out_name), args.scale)
        if res is None:
            print(f"Could not open {src}")
            continue
        n, fps, w, h = res
        print(f"{out_name}: {n} frames, {w}x{h} @ {fps:.0f} FPS, weather {c['weather']}")
        rows.append([out_name, c["name"], c["label"], c["weather"], n, "", ""])

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "original_clip", "traffic_class", "weather", "frames", "gt_dir_1", "gt_dir_2"])
        w.writerows(rows)
    print(f"Manifest: {manifest} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
