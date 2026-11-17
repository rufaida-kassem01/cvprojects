"""Save the frames around a counting event so you can inspect it.

Frame numbers in events.csv start at 1 for the first frame.

Usage (from the repo root):
    python tools/show_event.py --video outputs/s_count.mp4 --frame 76 --x 1172 --y 457
Saves images to outputs/event_<frame>/ with a red circle at the crossing point.
"""
import argparse
import os

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="annotated video (shows IDs) or the original")
    p.add_argument("--frame", type=int, required=True, help="frame number from events.csv (1-based)")
    p.add_argument("--x", type=int, required=True)
    p.add_argument("--y", type=int, required=True)
    p.add_argument("--span", type=int, default=6, help="frames to save before and after")
    p.add_argument("--scale", type=float, default=0.6, help="resize factor for the saved images")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    first = max(1, args.frame - args.span)
    last = min(total, args.frame + args.span)
    out_dir = f"outputs/event_{args.frame}"
    os.makedirs(out_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)  # OpenCV is 0-based
    for n in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        if n == args.frame:
            cv2.circle(img, (args.x, args.y), 25, (0, 0, 255), 4)
            cv2.putText(img, "EVENT", (args.x - 120, args.y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale)
        cv2.putText(img, f"frame {n}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.imwrite(os.path.join(out_dir, f"frame_{n:03d}.jpg"), img)

    cap.release()
    print(f"Saved frames {first}-{last} to {out_dir}")


if __name__ == "__main__":
    main()
