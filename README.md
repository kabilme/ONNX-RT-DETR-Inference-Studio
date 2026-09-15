# RT-DETR ONNX Real-Time Inference Application

A high-performance Python application to run object detection inference on `best.onnx` (Ultralytics RT-DETR-L model trained for `towerlamp` detection). Supports both image and video/webcam inputs with configurable inference metrics, device acceleration (CPU/DirectML GPU), and an interactive Web UI.

---

## ⚡ Key Features

- **Multi-Input Support**: Processes images (`.jpg`, `.png`, `.webp`), videos (`.mp4`, `.avi`, `.mov`), image folders, or live webcams/streams.
- **Configurable Inference Metrics**:
  - Confidence Threshold (`--conf`): Filter detections by minimum confidence score.
  - IoU Threshold (`--iou`): Non-Maximum Suppression overlap threshold.
  - Model Input Dimension (`--imgsz`): Dynamic resolution scaling (default: `640x640`).
  - Device Acceleration (`--device`): Supports `auto`, `cpu`, `dml` (Windows DirectML GPU), or `cuda`.
- **Latency & Performance Tracking**:
  - Granular latency measurement: Preprocessing, ONNX inference, Postprocessing, and Total latency in milliseconds.
  - Real-time FPS calculation.
  - Heads-Up Display (HUD) overlay directly rendered onto images and video frames.
  - Full metrics export to JSON for benchmark analysis.
- **Dual Interfaces**:
  - **CLI (`infer.py`)**: Command-line tool with progress bars and rich summary tables.
  - **Web Studio (`app.py`)**: Interactive Streamlit web dashboard with sliders, KPI cards, and download buttons.
- **Isolated Environment**: All dependencies run cleanly inside the `.venv` virtual environment.

---

## 🚀 Quick Start

### 1. Activate the Virtual Environment
```bash
.venv\Scripts\activate
```

### 2. Run Inference via CLI (`infer.py`)

#### Image Inference
```bash
python infer.py --source sample_test.jpg --conf 0.40 --save --export-metrics metrics.json
```

#### Video Inference
```bash
python infer.py --source sample_test.mp4 --conf 0.40 --save
```

#### Live Display with Interactive Keys
```bash
python infer.py --source sample_test.mp4 --show
```
*Keyboard shortcuts during live display:*
- `Q` or `ESC`: Exit
- `SPACE`: Pause / Resume
- `[` / `]`: Decrease / Increase confidence threshold by 0.05 live
- `H`: Toggle HUD overlay

#### Directory of Images
```bash
python infer.py --source path/to/images/ --conf 0.45 --save
```

#### Live Webcam
```bash
python infer.py --source 0 --show
```

---

## 🌐 Run Interactive Web Studio (`app.py`)

Launch the web application:
```bash
streamlit run app.py
```
Or simply double-click `run_app.bat`.

Then open your browser at `http://localhost:8501`.

---

## ⚙️ CLI Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source`, `-s` | `str` | *Required* | Path to image, video, directory, or webcam index (`0`) |
| `--model`, `-m` | `str` | `best.onnx` | Path to ONNX model file |
| `--conf`, `-c` | `float` | `0.45` | Confidence detection threshold (0.01 - 1.0) |
| `--iou`, `-i` | `float` | `0.45` | IoU / NMS threshold for duplicate box suppression |
| `--imgsz` | `int` | `640` | Input dimension for ONNX model |
| `--device`, `-d` | `str` | `auto` | `auto`, `cpu`, `dml` (DirectML GPU on Windows), `cuda` |
| `--save` | `flag` | `False` | Save annotated images/videos to output directory |
| `--output`, `-o` | `str` | `runs/detect` | Output directory root |
| `--show` | `flag` | `False` | Display live interactive OpenCV window |
| `--no-hud` | `flag` | `False` | Disable HUD metrics card overlay |
| `--thickness` | `int` | `2` | Bounding box line thickness |
| `--classes`, `--names` | `str` | `None` | Custom comma-separated classes (e.g. `'helmet'` or `'person,car'`) or path to classes text file to override model metadata |
| `--export-metrics` | `str` | `None` | Save metrics report to specified JSON file |

---

## 📁 Project Structure

```
d:\rt_inferance\
├── .venv\                  # Isolated Python 3.13 virtual environment
├── best.onnx               # Trained RT-DETR-L object detection model (towerlamp)
├── model_engine.py         # ONNX Runtime detector with pre/post-processing & metrics
├── visualizer.py           # Bounding box renderer and translucent HUD metrics card
├── infer.py                # Command-Line Interface (CLI) application
├── app.py                  # Streamlit Web Studio application
├── requirements.txt        # Pinned dependencies
├── run_cli.bat             # Batch launcher for CLI
├── run_app.bat             # Batch launcher for Web Studio
├── sample_test.jpg         # Sample test image
├── sample_test.mp4         # Sample test video
└── README.md               # Documentation
```
