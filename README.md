# RT-DETR ONNX Real-Time Inference Application

A high-performance Python application to run real-time object detection inference with ONNX Runtime (supports RT-DETR and YOLO ONNX models). Features configurable inference metrics, device acceleration (CPU, Windows DirectML GPU, or CUDA), Heads-Up Display (HUD) performance tracking, and an interactive Streamlit Web Studio.

---

## ⚡ Key Features

- **Multi-Input Support**: Processes images (`.jpg`, `.png`, `.webp`), videos (`.mp4`, `.avi`, `.mov`), image folders, or live webcams/streams.
- **Configurable Inference Metrics**:
  - Confidence Threshold (`--conf`): Filter detections by minimum confidence score.
  - IoU Threshold (`--iou`): Non-Maximum Suppression overlap threshold.
  - Model Input Dimension (`--imgsz`): Dynamic resolution scaling (default: `640x640`).
  - Device Acceleration (`--device`): Supports `auto`, `cpu`, `dml` (Windows DirectML GPU), or `cuda`.
- **Latency & Performance Tracking**:
  - Granular latency breakdown: Preprocessing, ONNX inference, Postprocessing, and Total latency in milliseconds.
  - Real-time FPS calculation.
  - Heads-Up Display (HUD) overlay directly rendered onto images and video frames.
  - Full metrics export to JSON for benchmark analysis.
- **Dual Interfaces**:
  - **CLI (`infer.py`)**: Command-line tool with progress bars and rich summary tables.
  - **Web Studio (`app.py`)**: Interactive Streamlit web dashboard with sliders, KPI cards, and download buttons.

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/kabilme/ONNX-RT-DETR-Inference-Studio.git
cd ONNX-RT-DETR-Inference-Studio

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Preparation

Place your trained ONNX model in the project root directory as `best.onnx`, or specify any model path using the `--model` argument:
```bash
# Example: If you have your own weights
copy path\to\your_model.onnx best.onnx
```

---

### 3. Run Inference via CLI (`infer.py`)

#### Image Inference
```bash
python infer.py --source sample_test.jpg --model best.onnx --conf 0.40 --save --export-metrics metrics.json
```

#### Video Inference
```bash
python infer.py --source sample_test.mp4 --model best.onnx --conf 0.40 --save
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

Launch the Streamlit web application:
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

You can select the model path, upload images or videos, adjust confidence and IoU sliders in real time, and download annotated media alongside latency reports.

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
| `--classes`, `--names` | `str` | `None` | Custom comma-separated classes or path to text file |
| `--export-metrics` | `str` | `None` | Save metrics report to specified JSON file |

---

## 📁 Project Structure

```
ONNX-RT-DETR-Inference-Studio/
├── app.py                  # Streamlit Web Studio application
├── infer.py                # Command-Line Interface (CLI) application
├── model_engine.py         # ONNX Runtime detector with pre/post-processing & metrics
├── visualizer.py           # Bounding box renderer and translucent HUD metrics card
├── requirements.txt        # Python dependencies
├── sample_test.jpg         # Sample test image
├── sample_test.mp4         # Sample test video
├── sample_img_metrics.json # Sample image benchmark output
├── sample_video_metrics.json# Sample video benchmark output
├── .gitignore              # Git ignore rules (weights, cache, runs)
└── README.md               # Documentation
```
