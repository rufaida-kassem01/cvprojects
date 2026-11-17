"""Detect, track and count vehicles crossing a line in a video.

Usage (from the repo root, venv active):
    python src/track_video.py --source data/traffic.mp4

First, check where the counting line sits (saves a preview image and exits):
    python src/track_video.py --source data/traffic.mp4 --preview-line

Outputs: an annotated video (--out), an events CSV (--events) and a text
report with settings, hardware, timing and counts (--report).
"""
import argparse
import csv
import datetime
import os
import platform
import statistics
import time

import cv2
import torch
import ultralytics
from ultralytics import YOLO

# COCO class ids: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]
WARMUP_FRAMES = 10  # first frames include CUDA/cuDNN init, excluded from timing
SHORT_TRACK_FRAMES = 15  # tracks shorter than this (0.5 s at 30 FPS) hint at fragmentation


# ---- Counting helpers -------------------------------------------------------
def side_of_line(p, a, b):
    """Sign of the cross product (b - a) x (p - a): which side of line a->b the point is on."""
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return (v > 0) - (v < 0)


def _orient(p, q, r):
    v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (v > 0) - (v < 0)


def segments_intersect(p1, p2, a, b):
    """True if segment p1-p2 (the vehicle's movement) crosses segment a-b (the counting line)."""
    return _orient(p1, p2, a) != _orient(p1, p2, b) and _orient(a, b, p1) != _orient(a, b, p2)


def parse_line(text, width, height):
    """Parse 'x1,y1,x2,y2' given as fractions (0-1) of the frame size into pixel coordinates."""
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise SystemExit("--line needs four numbers: x1,y1,x2,y2 (fractions of the frame, e.g. 0,0.6,1,0.6)")
    x1, y1, x2, y2 = vals
    return (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="path to a video file")
    parser.add_argument("--weights", default="yolo11s.pt", help="YOLO weights (downloaded on first use)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="tracker config: a built-in name or the path to your own yaml")
    parser.add_argument("--line", default="0,0.6,1,0.6",
                        help="counting line as x1,y1,x2,y2 fractions of the frame (default: horizontal at 60%% height)")
    parser.add_argument("--preview-line", action="store_true",
                        help="save outputs/line_preview.jpg with the line drawn, then exit")
    parser.add_argument("--out", default="outputs/annotated.mp4")
    parser.add_argument("--events", default="outputs/events.csv", help="events CSV path (empty string to skip)")
    parser.add_argument("--report", default="eval/baseline_week1.txt",
                        help="text report path (empty string to skip)")
    parser.add_argument("--show", action="store_true", help="show a live preview window (press q to quit)")
    args = parser.parse_args()

    # Read the source video properties so the output matches
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = cap.read()
    cap.release()

    line_a, line_b = parse_line(args.line, width, height)

    if args.preview_line:
        preview = first_frame.copy()
        cv2.line(preview, line_a, line_b, (0, 255, 255), 3)
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/line_preview.jpg", preview)
        print(f"Line {line_a} -> {line_b} on a {width}x{height} frame.")
        print("Saved outputs/line_preview.jpg. Adjust with --line x1,y1,x2,y2 (fractions of the frame).")
        return

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Device: {'GPU ' + device_name if use_gpu else 'CPU'}")
    print(f"Input: {width}x{height} @ {fps_in:.1f} FPS, {total_frames} frames")
    print(f"Counting line: {line_a} -> {line_b}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    model = YOLO(args.weights)
    unique_ids = set()
    first_seen, last_seen = {}, {}  # track id -> frame numbers, for track-length stats
    last_pos, last_side = {}, {}    # track id -> last centre point / last non-zero side of the line
    counted = set()                 # each track id is counted at most once
    counts = {"dir_1": 0, "dir_2": 0}
    events = []
    frames = 0

    # Per-stage timing (ms), measured after warm-up
    speed_sum = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    draw_write_s = 0.0
    t_start = time.perf_counter()
    t_after_warmup = None

    # stream=True yields results one frame at a time (low memory)
    results = model.track(
        source=args.source,
        stream=True,
        tracker=args.tracker,
        classes=VEHICLE_CLASSES,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        persist=True,
        verbose=False,
    )

    for result in results:
        frames += 1

        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            centres = result.boxes.xywh[:, :2].tolist()
            classes = result.boxes.cls.int().tolist()
            for tid, (cx, cy), cls in zip(ids, centres, classes):
                unique_ids.add(tid)
                first_seen.setdefault(tid, frames)
                last_seen[tid] = frames
                point = (cx, cy)
                side = side_of_line(point, line_a, line_b)
                prev_point = last_pos.get(tid)
                prev_side = last_side.get(tid)
                if (prev_point is not None and prev_side is not None and side != 0
                        and side != prev_side and tid not in counted
                        and segments_intersect(prev_point, point, line_a, line_b)):
                    direction = "dir_1" if prev_side < 0 and side > 0 else "dir_2"
                    counts[direction] += 1
                    counted.add(tid)
                    events.append((frames, round(frames / fps_in, 2), tid, result.names[cls], direction,
                                   round(cx), round(cy)))
                last_pos[tid] = point
                if side != 0:
                    last_side[tid] = side

        t0 = time.perf_counter()
        annotated = result.plot()  # draws boxes, class names and track IDs
        cv2.line(annotated, line_a, line_b, (0, 255, 255), 3)
        label = f"dir_1: {counts['dir_1']}   dir_2: {counts['dir_2']}   total: {counts['dir_1'] + counts['dir_2']}"
        cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        writer.write(annotated)
        t_draw = time.perf_counter() - t0

        if frames > WARMUP_FRAMES:
            for k in speed_sum:
                speed_sum[k] += result.speed[k]
            draw_write_s += t_draw
        if frames == WARMUP_FRAMES:
            t_after_warmup = time.perf_counter()

        if args.show:
            cv2.imshow("streamsight", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    t_end = time.perf_counter()
    writer.release()
    cv2.destroyAllWindows()

    if args.events:
        os.makedirs(os.path.dirname(args.events) or ".", exist_ok=True)
        with open(args.events, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "track_id", "class", "direction", "x", "y"])
            w.writerows(events)

    # ---- Build the report (printed and optionally saved) ----
    total_s = t_end - t_start
    lengths = [last_seen[t] - first_seen[t] + 1 for t in first_seen]
    short = sum(1 for n in lengths if n < SHORT_TRACK_FRAMES)
    lines = [
        "StreamSight report",
        f"Date:       {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"Hardware:   {device_name} | CPU: {platform.processor() or platform.machine()}",
        f"Software:   Python {platform.python_version()} | torch {torch.__version__} "
        f"(CUDA {torch.version.cuda}) | ultralytics {ultralytics.__version__}",
        f"Model:      {args.weights} | imgsz {args.imgsz} | conf {args.conf} | "
        f"classes {VEHICLE_CLASSES}",
        f"Tracker:    {args.tracker}",
        f"Input:      {os.path.basename(args.source)} | {width}x{height} @ {fps_in:.1f} FPS | {total_frames} frames",
        f"Line:       {line_a} -> {line_b} (pixels); each track ID is counted at most once",
        "",
        f"Processed {frames} frames in {total_s:.1f}s -> {frames / total_s:.1f} FPS (including warm-up)",
    ]

    measured = frames - WARMUP_FRAMES
    if t_after_warmup is not None and measured > 0:
        steady_s = t_end - t_after_warmup
        steady_ms = steady_s / measured * 1000
        lines.append(f"Steady state (after {WARMUP_FRAMES} warm-up frames): {measured / steady_s:.1f} FPS end to end "
                     f"({steady_ms:.1f} ms per frame)")
        lines.append("Average per-frame time after warm-up (ms):")
        for k, v in speed_sum.items():
            lines.append(f"  {k:<20} {v / measured:6.1f}")
        draw_ms = draw_write_s / measured * 1000
        other = steady_ms - sum(speed_sum.values()) / measured - draw_ms
        lines.append(f"  {'draw+write':<20} {draw_ms:6.1f}")
        lines.append(f"  {'decode+track+count':<20} {other:6.1f}")

    lines.append("")
    lines.append(f"Line crossings: dir_1 {counts['dir_1']} | dir_2 {counts['dir_2']} | total {sum(counts.values())}")
    lines.append("  (dir_1 = crossing from the negative to the positive side of the line a->b; "
                 "for the default horizontal line that is moving down the frame, dir_2 is moving up)")
    lines.append(f"Unique track IDs seen: {len(unique_ids)} (not a vehicle count; IDs can switch)")
    if lengths:
        lines.append(f"Track length (frames): median {statistics.median(lengths):.0f}, "
                     f"{short} of {len(lengths)} tracks shorter than {SHORT_TRACK_FRAMES} frames")
    lines.append(f"Annotated video: {args.out}")
    if args.events:
        lines.append(f"Events CSV: {args.events} ({len(events)} events)")

    print()
    print("\n".join(lines))

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()