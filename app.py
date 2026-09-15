"""
Streamlit Web Application for Real-Time ONNX Inference.
Features interactive controls for confidence, IoU thresholds, device acceleration,
live image/video inference, KPI metric cards, and downloadable annotated media & reports.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from model_engine import Detection, InferenceMetrics, ONNXDetector
from visualizer import Visualizer

# Page Configuration
st.set_page_config(
    page_title="ONNX RT-DETR Inference Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #38bdf8;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(
    model_path: str,
    imgsz: int,
    device: str,
    mtime: float,
    custom_classes: Optional[str] = None,
) -> ONNXDetector:
    """Cached loader for ONNX detector with auto-reload on model file change."""
    return ONNXDetector(
        model_path=model_path,
        conf_threshold=0.45,
        iou_threshold=0.45,
        input_size=(imgsz, imgsz),
        device=device,
        custom_classes=custom_classes if custom_classes else None,
    )


def create_sample_image() -> np.ndarray:
    """Generate a high-contrast synthetic test image for detection testing."""
    img = np.ones((640, 800, 3), dtype=np.uint8) * 40

    # Draw simulated object / person with helmet
    # Head & helmet
    cv2.circle(img, (400, 260), 45, (0, 200, 255), -1)  # Yellow helmet
    cv2.circle(img, (400, 260), 48, (255, 255, 255), 2)
    # Face
    cv2.circle(img, (400, 290), 30, (180, 190, 220), -1)
    # Torso
    cv2.rectangle(img, (340, 330), (460, 520), (30, 80, 180), -1)
    # Reflective vest strips
    cv2.rectangle(img, (350, 380), (450, 410), (0, 230, 70), -1)

    cv2.putText(
        img, "Industrial Safety Test Image", (180, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA
    )
    return img


def main():
    st.title("⚡ ONNX RT-DETR Inference Studio")
    st.caption("High-performance object detection inference with configurable metrics on `best.onnx`")

    # Sidebar: Model and Configuration
    with st.sidebar:
        st.header("⚙️ Model & Metrics Config")

        model_path = st.text_input("Model Path", value="best.onnx")
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            st.stop()

        mtime = os.path.getmtime(model_path)

        device_option = st.selectbox(
            "Execution Device",
            options=["auto", "cpu", "dml", "cuda"],
            index=0,
            help="'dml' enables Windows DirectML GPU acceleration; 'cpu' uses CPUExecutionProvider.",
        )

        imgsz_option = st.select_slider(
            "Model Input Dimension",
            options=[320, 480, 640, 800],
            value=640,
            help="Higher values increase detection accuracy for small objects at the cost of latency.",
        )

        custom_classes_input = st.text_input(
            "Custom Class Names (optional)",
            value="",
            help="Leave empty to auto-detect from model metadata, or enter comma-separated classes (e.g. 'helmet' or 'person,helmet').",
        ).strip()

        # Load detector (mtime ensures auto-reload when best.onnx is replaced)
        try:
            detector = load_detector(
                model_path,
                imgsz_option,
                device_option,
                mtime,
                custom_classes_input if custom_classes_input else None,
            )
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
            st.stop()

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Active Provider</div>
                <div style="font-weight: 600; color: #10b981;">{detector.active_provider}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Active Classes</div>
                <div style="font-weight: 600; color: #38bdf8;">{list(detector.names.values())}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄 Force Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.subheader("🎯 Detection Thresholds")
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.00,
            value=0.45,
            step=0.02,
            help="Minimum confidence score required for an object to be detected.",
        )

        iou_thresh = st.slider(
            "IoU / NMS Threshold",
            min_value=0.01,
            max_value=1.00,
            value=0.45,
            step=0.05,
            help="Intersection-over-Union threshold for filtering duplicate overlapping boxes.",
        )

        detector.set_thresholds(conf=conf_thresh, iou=iou_thresh)

        st.subheader("🎨 Visual Overlay")
        show_hud = st.checkbox("Show HUD Metrics Card", value=True)
        show_labels = st.checkbox("Show Class Labels", value=True)
        show_conf = st.checkbox("Show Confidence %", value=True)
        thickness = st.slider("Box Line Thickness", min_value=1, max_value=6, value=2)

        visualizer = Visualizer(
            show_hud=show_hud,
            show_labels=show_labels,
            show_conf=show_conf,
            box_thickness=thickness,
        )

    # Main Area Tabs
    tab_image, tab_video, tab_batch = st.tabs(["📷 Image Inference", "🎥 Video Inference", "📁 Batch Folder"])

    # -------------------------------------------------------------
    # TAB 1: IMAGE INFERENCE
    # -------------------------------------------------------------
    with tab_image:
        col_ctrl, col_sample = st.columns([3, 1])
        with col_ctrl:
            uploaded_image = st.file_uploader(
                "Upload Image (JPG, PNG, WEBP, BMP)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )
        with col_sample:
            st.write("&nbsp;")
            use_sample = st.button("Use Sample Test Image", use_container_width=True)

        input_cv2_image = None
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            input_cv2_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif use_sample:
            input_cv2_image = create_sample_image()

        if input_cv2_image is not None:
            # Run Inference
            result = detector.predict(input_cv2_image)
            annotated_bgr = visualizer.render(
                result.original_image,
                result.detections,
                result.metrics,
                model_name=os.path.basename(detector.model_path),
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
            )

            # Display KPI Metric Cards
            m = result.metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">FPS</div><div class="metric-val">{m.fps:.1f}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Total Latency</div><div class="metric-val">{m.total_ms:.1f} ms</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">ONNX Inference</div><div class="metric-val">{m.inference_ms:.1f} ms</div></div>""",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Pre / Post Latency</div><div class="metric-val">{m.preprocess_ms:.1f} / {m.postprocess_ms:.1f} ms</div></div>""",
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Detections Found</div><div class="metric-val">{m.detection_count}</div></div>""",
                    unsafe_allow_html=True,
                )

            # Image Comparison
            col_orig, col_ann = st.columns(2)
            with col_orig:
                st.subheader("Original Input")
                st.image(cv2.cvtColor(result.original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_ann:
                st.subheader("Inference Result + HUD")
                st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Detection Table & Downloads
            if result.detections:
                st.subheader(f"📋 Detections List ({len(result.detections)})")
                table_data = [
                    {
                        "Index": i + 1,
                        "Class": d.class_name,
                        "Confidence": f"{d.confidence * 100:.2f}%",
                        "Bounding Box [x1, y1, x2, y2]": [round(c, 1) for c in d.bbox],
                    }
                    for i, d in enumerate(result.detections)
                ]
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.info(f"No objects detected with confidence >= {conf_thresh:.2f}. Try lowering the confidence slider in the sidebar.")

            # Download Buttons
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                success, enc_img = cv2.imencode(".jpg", annotated_bgr)
                if success:
                    st.download_button(
                        label="⬇️ Download Annotated Image",
                        data=enc_img.tobytes(),
                        file_name="annotated_detection.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )
            with dl_col2:
                metrics_json = json.dumps(
                    {
                        "metrics": m.to_dict(),
                        "detections": [d.to_dict() for d in result.detections],
                    },
                    indent=2,
                )
                st.download_button(
                    label="⬇️ Download Metrics JSON Report",
                    data=metrics_json,
                    file_name="inference_metrics.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("Upload an image above or click 'Use Sample Test Image' to test inference.")

    # -------------------------------------------------------------
    # TAB 2: VIDEO INFERENCE
    # -------------------------------------------------------------
    with tab_video:
        st.subheader("🎥 Video Inference & Metrics")
        uploaded_video = st.file_uploader(
            "Upload Video File (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            st.write(f"**Resolution**: {width}x{height} | **Frames**: {total_frames} | **FPS**: {video_fps:.1f}")

            max_frames_to_process = st.number_input(
                "Max Frames to Process (0 = All)",
                min_value=0,
                max_value=total_frames,
                value=min(total_frames, 150),
                step=25,
            )

            if st.button("🚀 Run Video Inference", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                preview_placeholder = st.empty()

                cap = cv2.VideoCapture(tfile.name)
                limit = total_frames if max_frames_to_process == 0 else max_frames_to_process

                out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_temp.close()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_temp.name, fourcc, video_fps, (width, height))

                frame_idx = 0
                all_metrics: List[Dict] = []
                total_dets = 0

                start_time = time.time()
                while cap.isOpened() and frame_idx < limit:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    result = detector.predict(frame)
                    all_metrics.append(result.metrics.to_dict())
                    total_dets += len(result.detections)

                    annotated = visualizer.render(
                        result.original_image,
                        result.detections,
                        result.metrics,
                        model_name=os.path.basename(detector.model_path),
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh,
                        extra_info=f"Frame: {frame_idx}/{limit}",
                    )

                    writer.write(annotated)

                    # Update preview every few frames
                    if frame_idx % 5 == 0 or frame_idx == limit:
                        preview_placeholder.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=f"Processing Frame {frame_idx}/{limit}",
                            use_container_width=True,
                        )

                    progress_bar.progress(float(frame_idx / limit))
                    status_text.text(f"Processed {frame_idx}/{limit} frames... ({result.metrics.fps:.1f} FPS)")

                cap.release()
                writer.release()
                elapsed = time.time() - start_time

                st.success(f"Processed {frame_idx} frames in {elapsed:.2f}s (Average: {frame_idx / elapsed:.1f} FPS)!")

                # Video Performance Summary
                if all_metrics:
                    fps_vals = [m["fps"] for m in all_metrics if m["fps"] > 0]
                    total_ms_vals = [m["total_ms"] for m in all_metrics]
                    infer_ms_vals = [m["inference_ms"] for m in all_metrics]

                    vc1, vc2, vc3, vc4 = st.columns(4)
                    vc1.metric("Average FPS", f"{np.mean(fps_vals):.1f} FPS")
                    vc2.metric("Mean Latency", f"{np.mean(total_ms_vals):.1f} ms")
                    vc3.metric("Mean Inference", f"{np.mean(infer_ms_vals):.1f} ms")
                    vc4.metric("Total Detections", str(total_dets))

                    # Line Chart of Latency & FPS
                    df_metrics = pd.DataFrame(
                        {
                            "Frame": list(range(1, len(all_metrics) + 1)),
                            "Inference Latency (ms)": infer_ms_vals,
                            "Total Latency (ms)": total_ms_vals,
                            "FPS": fps_vals,
                        }
                    )
                    st.line_chart(df_metrics.set_index("Frame")[["Inference Latency (ms)", "Total Latency (ms)"]])

                    # Video Download
                    with open(out_temp.name, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=video_bytes,
                        file_name="annotated_video.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

    # -------------------------------------------------------------
    # TAB 3: BATCH FOLDER
    # -------------------------------------------------------------
    with tab_batch:
        st.subheader("📁 Batch Folder Inference")
        folder_path = st.text_input("Local Folder Path containing images:", value="")

        if st.button("Scan & Run Batch Inference") and folder_path:
            if not os.path.isdir(folder_path):
                st.error("Invalid directory path.")
            else:
                image_files = []
                for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    image_files.extend(Path(folder_path).glob(f"*{ext}"))
                    image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

                if not image_files:
                    st.warning(f"No image files found in {folder_path}.")
                else:
                    st.info(f"Found {len(image_files)} images. Running batch inference...")
                    batch_bar = st.progress(0.0)
                    batch_metrics = []
                    total_batch_dets = 0

                    out_batch_dir = Path("runs") / "batch_predict"
                    out_batch_dir.mkdir(parents=True, exist_ok=True)

                    for i, img_p in enumerate(image_files):
                        img = cv2.imread(str(img_p))
                        if img is not None:
                            res = detector.predict(img)
                            batch_metrics.append(res.metrics.to_dict())
                            total_batch_dets += len(res.detections)
                            ann = visualizer.render(
                                res.original_image,
                                res.detections,
                                res.metrics,
                                conf_thresh=conf_thresh,
                                iou_thresh=iou_thresh,
                            )
                            cv2.imwrite(str(out_batch_dir / img_p.name), ann)
                        batch_bar.progress((i + 1) / len(image_files))

                    st.success(
                        f"Batch complete! Processed {len(image_files)} images with {total_batch_dets} total detections."
                    )
                    st.write(f"Results saved to: `{out_batch_dir.resolve()}`")


if __name__ == "__main__":
    main()
