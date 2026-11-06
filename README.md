# StreamSight: Real-Time Multi-Stream Video Analytics

> From raw video to counted, tracked, alerted events, in real time.

Real-time multi-stream video analytics: vehicle detection, tracking, line-crossing counts and zone-intrusion alerts with YOLO + ByteTrack.

<!-- Add a 10-20 second demo GIF here (annotated video with counts and IDs) -->
![demo](docs/demo.gif)

## What it does

- Detects vehicles with a pretrained YOLO model and tracks them across frames with ByteTrack.
- Counts vehicles crossing a virtual line, per direction.
- Raises alerts when an object enters a defined zone, and saves a snapshot of each event.
- Runs several video streams at once and reports per-stream performance.
- Shows counts over time and recent alerts in a small dashboard.

## Architecture

<!-- Add a simple diagram: video source -> decode (thread) -> queue -> detect -> track -> counting/alerts -> storage -> dashboard -->
![architecture](docs/architecture.png)

## Results

All numbers below are measured on my own labeled test clips. Hardware: `<GPU model, CPU, RAM>`.

### Counting accuracy

| Clip | Conditions | Ground truth | Predicted | Error |
|------|-----------|--------------|-----------|-------|
| `<clip 1>` | `<day / clear>` | `<n>` | `<n>` | `<%>` |
| `<clip 2>` | `<crowded>` | `<n>` | `<n>` | `<%>` |
| `<clip 3>` | `<night / rain>` | `<n>` | `<n>` | `<%>` |

### Performance

| Setup | Streams | FPS per stream | Latency per frame |
|-------|---------|----------------|-------------------|
| PyTorch baseline | 1 | `<fps>` | `<ms>` |
| + threaded decoding | 1 | `<fps>` | `<ms>` |
| ONNX | 1 | `<fps>` | `<ms>` |
| ONNX | `<2-4>` | `<fps>` | `<ms>` |

Per-stage timing (decode / detect / track / draw): `<fill in from profiling>`

## Failure analysis

What goes wrong, why, and what I would do about it.

| Failure case | Cause | Possible fix |
|--------------|-------|--------------|
| `<ID switches in occlusion>` | `<cause>` | `<fix>` |
| `<double counting near the line>` | `<cause>` | `<fix>` |
| `<missed detections at night>` | `<cause>` | `<fix>` |

## Getting started

```bash
git clone https://github.com/<your-username>/streamsight.git
cd streamsight
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run on a video file:

```bash
python -m src.pipeline --config configs/default.yaml --source path/to/video.mp4
```

Run with Docker:

```bash
docker build -t streamsight .
docker run --rm streamsight
```

## Configuration

Zones, counting lines, thresholds and stream sources are set in `configs/`. Example:

```yaml
sources:
  - path/to/video1.mp4
model:
  weights: yolo-weights-file
  confidence: 0.35
counting_line:
  start: [0, 400]
  end: [1280, 400]
zones:
  - name: restricted-area
    polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]
```

## Project structure

```
streamsight/
├── README.md
├── configs/          # zones, thresholds, stream sources
├── src/
│   ├── detect.py
│   ├── track.py
│   ├── counting.py
│   ├── alerts.py
│   ├── pipeline.py   # queues, threads, multi-stream
│   └── dashboard.py
├── eval/             # ground truth, scripts, results
├── tests/
├── Dockerfile
└── requirements.txt
```

## Limitations and next steps

- `<known limitations, stated honestly>`
- `<what you would improve next: e.g. fine-tuning on local data, cross-camera re-identification, heatmaps>`

## Data and licenses

- Test videos: `<source and license>`
- Model weights: `<source and license>`
- Code license: `<e.g. MIT>`