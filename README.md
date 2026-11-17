# StreamSight: Real-Time Multi-Stream Video Analytics

> From raw video to counted, tracked, alerted events, in real time.

Real-time multi-stream video analytics: vehicle detection, tracking, line-crossing counts and zone-intrusion alerts with YOLO + ByteTrack.

**Status: work in progress.** The single-stream detection and tracking baseline is done and measured. Counting, evaluation, multi-stream support and alerts are next (see the roadmap).

## What works today

- Detects vehicles (car, motorcycle, bus, truck) with a pretrained YOLO model.
- Tracks them across frames with ByteTrack, so each vehicle gets an ID.
- Saves an annotated video and a text report with the settings, hardware and per-stage timing.

## Baseline results (Week 1)

Single stream, pretrained `yolo11n.pt`, 640 px inference, confidence 0.35, ByteTrack, PyTorch (no export or optimization yet).

**Hardware and software:** NVIDIA GeForce GTX 1660 Ti Max-Q (6 GB, laptop), Python 3.14, PyTorch 2.15 (CUDA 13.4 build).
**Test clip:** 1280x720 at 30 FPS, 201 frames (about 7 seconds), fixed elevated camera on a multi-lane road.

| Metric | Result |
|--------|--------|
| Steady-state speed (after 10 warm-up frames) | about 31 FPS end to end (about 32 ms per frame) |
| Speed including warm-up | 5.5 to 12.4 FPS (the first frames pay for CUDA initialization; shorter on the second run) |

Per-frame time breakdown (ms, second run):

| Stage | ms |
|-------|----|
| Preprocess | 1.8 |
| Inference | 14.4 |
| Postprocess | 2.4 |
| Draw boxes + write video | 10.6 |
| Decode + tracking + other | 2.6 |

The detector is not the bottleneck: about a third of the frame time goes to drawing and encoding the annotated output, which a deployed pipeline would not do on every frame. Measuring a headless mode is part of Week 3.

The full report from each run is saved to `eval/baseline_week1.txt`.

## Failure analysis so far

These are observations from one clip and will be turned into measured results during the evaluation phase.

| Observation | Likely cause | To investigate |
|-------------|--------------|----------------|
| In a sample frame, only about 4 vehicles have boxes while several other clearly visible vehicles (vans, a gray sedan in the foreground) have none | Heavy motion blur, close camera, very large vehicles, small `yolo11n` model, confidence threshold 0.35 | Larger model (`yolo11s`/`yolo11m`), lower threshold, a clip with less blur |
| 103 unique track IDs over a 7-second clip, with IDs already above 30 early in the clip | ID switches and fragmented tracks when detections drop out | Track-length statistics, tracker parameters, ground-truth comparison |

## Roadmap

- [x] Baseline: YOLO detection + ByteTrack on one video, with timing report
- [ ] Counting line with per-direction counts and an event CSV
- [ ] Evaluation on labeled clips (count error, failure cases in day, crowded and night or rain conditions)
- [ ] Real-time performance: threaded decoding, headless mode, ONNX export, FPS comparison
- [ ] Multi-stream processing (2 to 4 streams) with per-stream FPS
- [ ] Zone-intrusion alerts with saved snapshots
- [ ] Dashboard, configuration file, Docker, tests

## Getting started

```bash
git clone https://github.com/rufaida-kassem01/streamsight.git
cd streamsight
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
```

Install PyTorch first, using the command from https://pytorch.org/get-started/locally/ that matches your CUDA version (or the CPU build), then the rest:

```bash
pip install ultralytics opencv-python
```

Run on a video file:

```bash
python src/track_video.py --source data/traffic.mp4
```

Useful options: `--weights yolo11s.pt` (larger model), `--conf 0.25`, `--imgsz 960`, `--show` (live preview, press `q` to quit), `--out`, `--report`.

## Project structure

```
streamsight/
├── README.md
├── src/
│   └── track_video.py    # detection + tracking baseline with timing report
├── eval/                 # reports and, later, ground truth and results
├── requirements.txt
└── .gitignore            # excludes the venv, videos and model weights
```

## Data and licenses

- Test video: stock footage, not included in the repo (`data/` is git-ignored). Source and license: *add the clip's source and license here*.
- Model weights: pretrained Ultralytics YOLO weights, downloaded on first run.
- Code license: *add a license*.