"""
CLI Application for Real-Time ONNX Inference on Images and Videos.
Runs object detection on best.onnx with configurable metrics, thresholds,
device acceleration, HUD overlay, live interactive display, and metrics export.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from model_engine import Detection, InferenceMetrics, ONNXDetector
from visualizer import Visualizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ONNX object detection inference on images and videos with configurable metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Path to image, video file, directory of images, or webcam index (e.g. '0').",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="best.onnx",
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.45,
        help="Confidence threshold for detections (0.01 - 1.0).",
    )
    parser.add_argument(
        "--iou",
        "-i",
        type=float,
        default=0.45,
        help="IoU / NMS threshold for duplicate box filtering (0.01 - 1.0).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input dimension size (square resolution) for model.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "dml", "cuda"],
        help="Execution provider: 'cpu', 'dml' (DirectML GPU on Windows), 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save annotated images or videos to output directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="runs/detect",
        help="Root output directory to save results.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display live OpenCV window during processing with interactive controls.",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        default=False,
        help="Disable the on-screen Heads-Up Display (HUD) metrics overlay.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        default=False,
        help="Hide class name labels on bounding boxes.",
    )
    parser.add_argument(
        "--no-conf",
        action="store_true",
        default=False,
        help="Hide confidence percentages on bounding boxes.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Bounding box line thickness in pixels.",
    )
    parser.add_argument(
        "--classes",
        "--names",
        type=str,
        default=None,
        help="Optional comma-separated class names (e.g. 'helmet' or 'person,car') or path to classes text file to override model metadata.",
    )
    parser.add_argument(
        "--export-metrics",
        type=str,
        default=None,
        help="File path to export JSON metrics report.",
    )
    return parser.parse_args()


def get_incremental_dir(base_dir: str, prefix: str = "exp") -> Path:
    """Create incremented directory like runs/detect/exp, exp2, exp3..."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        sub = base / f"{prefix}{n if n > 1 else ''}"
        if not sub.exists():
            sub.mkdir(parents=True, exist_ok=True)
            return sub
        n += 1


def process_image(
    image_path: str,
    detector: ONNXDetector,
    visualizer: Visualizer,
    save_dir: Optional[Path],
    show: bool,
) -> Dict:
    """Process a single image file."""
    image = cv2.imread(image_path)
    if image is None:
        console.print(f"[red]Failed to read image:[/red] {image_path}")
        return {}

    result = detector.predict(image)
    annotated = visualizer.render(
        result.original_image,
        result.detections,
        result.metrics,
        model_name=os.path.basename(detector.model_path),
        conf_thresh=detector.conf_threshold,
        iou_thresh=detector.iou_threshold,
    )

    if save_dir:
        save_path = save_dir / os.path.basename(image_path)
        cv2.imwrite(str(save_path), annotated)

    if show:
        cv2.imshow("ONNX Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "file": image_path,
        "metrics": result.metrics.to_dict(),
        "detections": [d.to_dict() for d in result.detections],
    }


def process_video_stream(
    source: Union[str, int],
    detector: ONNXDetector,
    visualizer: Visualizer,
    save_dir: Optional[Path],
    show: bool,
) -> Dict:
    """Process a video file, stream, or webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        console.print(f"[red]Error opening video source:[/red] {source}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live = total_frames <= 0

    writer = None
    if save_dir:
        src_name = "webcam.mp4" if isinstance(source, int) else Path(str(source)).stem + "_detected.mp4"
        save_path = save_dir / src_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, video_fps, (width, height))

    all_metrics: List[Dict] = []
    detection_counts_per_class: Dict[str, int] = {}
    total_detections = 0
    frame_idx = 0
    paused = False

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn() if not is_live else TextColumn("Frames: {task.completed}"),
        TimeElapsedColumn(),
        TimeRemainingColumn() if not is_live else TextColumn(""),
        console=console,
    )

    task_desc = "Processing Live Stream" if is_live else f"Processing Video ({width}x{height} @ {video_fps:.1f}fps)"
    task_id = progress.add_task(task_desc, total=total_frames if not is_live else None)

    if show:
        console.print(
            "[dim]Interactive Controls: [[bold]Q[/bold]/[bold]ESC[/bold]]: Quit | [[bold]SPACE[/bold]]: Pause | [[bold][[/bold]/[bold]][/bold]]: Adjust Conf ±0.05 | [[bold]H[/bold]]: Toggle HUD[/dim]"
        )

    with progress:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                result = detector.predict(frame)
                m = result.metrics

                all_metrics.append(m.to_dict())
                total_detections += len(result.detections)
                for det in result.detections:
                    detection_counts_per_class[det.class_name] = (
                        detection_counts_per_class.get(det.class_name, 0) + 1
                    )

                annotated = visualizer.render(
                    result.original_image,
                    result.detections,
                    m,
                    model_name=os.path.basename(detector.model_path),
                    conf_thresh=detector.conf_threshold,
                    iou_thresh=detector.iou_threshold,
                    extra_info=f"Frame: {frame_idx}" + (f"/{total_frames}" if not is_live else ""),
                )

                if writer:
                    writer.write(annotated)

                progress.update(task_id, advance=1)

            if show:
                cv2.imshow("ONNX Inference Stream", annotated if not paused else frame)
                key = cv2.waitKey(1 if not paused else 50) & 0xFF
                if key in (ord("q"), 27):  # Q or ESC
                    break
                elif key == ord(" "):  # Space
                    paused = not paused
                elif key == ord("["):  # Decrease conf
                    detector.set_thresholds(conf=max(0.05, detector.conf_threshold - 0.05))
                elif key == ord("]"):  # Increase conf
                    detector.set_thresholds(conf=min(0.95, detector.conf_threshold + 0.05))
                elif key == ord("h"):  # Toggle HUD
                    visualizer.show_hud = not visualizer.show_hud

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "source": str(source),
        "total_frames": frame_idx,
        "total_detections": total_detections,
        "class_counts": detection_counts_per_class,
        "frame_metrics": all_metrics,
    }


def print_summary_table(
    all_metrics: List[Dict],
    total_detections: int,
    class_counts: Dict[str, int],
    save_dir: Optional[Path],
    device: str,
):
    """Print an aesthetic summary table using Rich."""
    if not all_metrics:
        return

    fps_list = [m["fps"] for m in all_metrics if m["fps"] > 0]
    total_ms_list = [m["total_ms"] for m in all_metrics]
    infer_ms_list = [m["inference_ms"] for m in all_metrics]
    pre_ms_list = [m["preprocess_ms"] for m in all_metrics]
    post_ms_list = [m["postprocess_ms"] for m in all_metrics]

    avg_fps = np.mean(fps_list) if fps_list else 0.0
    avg_total = np.mean(total_ms_list) if total_ms_list else 0.0
    avg_infer = np.mean(infer_ms_list) if infer_ms_list else 0.0
    avg_pre = np.mean(pre_ms_list) if pre_ms_list else 0.0
    avg_post = np.mean(post_ms_list) if post_ms_list else 0.0

    table = Table(
        title="[bold cyan]Inference Performance & Detection Summary[/bold cyan]",
        title_style="bold cyan",
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Metric", style="dim", width=26)
    table.add_column("Value", style="bold green")

    table.add_row("Execution Provider", device)
    table.add_row("Frames / Images Processed", str(len(all_metrics)))
    table.add_row("Total Detections Found", str(total_detections))
    for cname, count in class_counts.items():
        table.add_row(f"  └─ Class: '{cname}'", f"{count} instances")

    table.add_row("Average FPS", f"{avg_fps:.1f} FPS")
    table.add_row("Average Total Latency", f"{avg_total:.2f} ms")
    table.add_row("Average Preprocessing", f"{avg_pre:.2f} ms")
    table.add_row("Average ONNX Inference", f"{avg_infer:.2f} ms")
    table.add_row("Average Postprocessing", f"{avg_post:.2f} ms")

    if save_dir:
        table.add_row("Results Saved In", str(save_dir))

    console.print()
    console.print(table)
    console.print()


def main():
    args = parse_args()

    # 1. Initialize Detector
    try:
        detector = ONNXDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            input_size=(args.imgsz, args.imgsz),
            device=args.device,
            custom_classes=args.classes,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize detector:[/bold red] {e}")
        sys.exit(1)

    header = Panel(
        f"[bold cyan]RT-DETR / ONNX Inference Engine[/bold cyan]\n"
        f"[dim]Model:[/dim] [green]{args.model}[/green]  "
        f"[dim]Device:[/dim] [yellow]{detector.active_provider}[/yellow]\n"
        f"[dim]Classes:[/dim] [cyan]{list(detector.names.values())}[/cyan]\n"
        f"[dim]Conf Thresh:[/dim] [magenta]{args.conf}[/magenta]  "
        f"[dim]IoU Thresh:[/dim] [magenta]{args.iou}[/magenta]",
        expand=False,
    )
    console.print(header)

    # 2. Initialize Visualizer
    visualizer = Visualizer(
        show_hud=not args.no_hud,
        show_labels=not args.no_labels,
        show_conf=not args.no_conf,
        box_thickness=args.thickness,
    )

    # 3. Setup output directory if saving
    save_dir = None
    if args.save or args.export_metrics:
        save_dir = get_incremental_dir(args.output, "exp")

    # 4. Determine input source type
    source = args.source
    is_cam = source.isdigit() or source.startswith("rtsp://") or source.startswith("http://")

    summary_metrics: List[Dict] = []
    total_detections = 0
    class_counts: Dict[str, int] = {}
    detailed_results = []

    if is_cam:
        cam_id = int(source) if source.isdigit() else source
        video_res = process_video_stream(cam_id, detector, visualizer, save_dir, args.show)
        summary_metrics = video_res.get("frame_metrics", [])
        total_detections = video_res.get("total_detections", 0)
        class_counts = video_res.get("class_counts", {})
        detailed_results = video_res

    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            console.print(f"[cyan]Processing image:[/cyan] {source}")
            img_res = process_image(source, detector, visualizer, save_dir, args.show)
            if img_res:
                summary_metrics = [img_res["metrics"]]
                total_detections = len(img_res["detections"])
                for d in img_res["detections"]:
                    class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
                detailed_results = [img_res]

        elif ext in VIDEO_EXTENSIONS:
            console.print(f"[cyan]Processing video:[/cyan] {source}")
            video_res = process_video_stream(source, detector, visualizer, save_dir, args.show)
            summary_metrics = video_res.get("frame_metrics", [])
            total_detections = video_res.get("total_detections", 0)
            class_counts = video_res.get("class_counts", {})
            detailed_results = video_res
        else:
            console.print(f"[bold red]Unsupported file extension:[/bold red] {ext}")
            sys.exit(1)

    elif os.path.isdir(source):
        console.print(f"[cyan]Processing image folder:[/cyan] {source}")
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(source, f"*{ext}")))
            files.extend(glob.glob(os.path.join(source, f"*{ext.upper()}")))

        if not files:
            console.print(f"[yellow]No images found in directory:[/yellow] {source}")
            sys.exit(0)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task(f"Processing {len(files)} images", total=len(files))
            for fpath in files:
                img_res = process_image(fpath, detector, visualizer, save_dir, args.show)
                if img_res:
                    summary_metrics.append(img_res["metrics"])
                    total_detections += len(img_res["detections"])
                    for d in img_res["detections"]:
                        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
                    detailed_results.append(img_res)
                prog.update(task, advance=1)

    else:
        console.print(f"[bold red]Source path does not exist:[/bold red] {source}")
        sys.exit(1)

    # 5. Print summary
    print_summary_table(
        summary_metrics,
        total_detections,
        class_counts,
        save_dir,
        detector.active_provider,
    )

    # 6. Export metrics
    export_path = args.export_metrics
    if not export_path and save_dir:
        export_path = str(save_dir / "metrics_report.json")

    if export_path:
        report = {
            "model": args.model,
            "device": detector.active_provider,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "img_size": args.imgsz,
            "total_items_processed": len(summary_metrics),
            "total_detections": total_detections,
            "class_distribution": class_counts,
            "summary_performance": {
                "avg_fps": round(float(np.mean([m["fps"] for m in summary_metrics if m["fps"] > 0])), 1)
                if summary_metrics
                else 0.0,
                "avg_total_ms": round(float(np.mean([m["total_ms"] for m in summary_metrics])), 2)
                if summary_metrics
                else 0.0,
                "avg_preprocess_ms": round(float(np.mean([m["preprocess_ms"] for m in summary_metrics])), 2)
                if summary_metrics
                else 0.0,
                "avg_inference_ms": round(float(np.mean([m["inference_ms"] for m in summary_metrics])), 2)
                if summary_metrics
                else 0.0,
                "avg_postprocess_ms": round(float(np.mean([m["postprocess_ms"] for m in summary_metrics])), 2)
                if summary_metrics
                else 0.0,
            },
            "detailed_data": detailed_results,
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"[bold green][OK] Detailed metrics exported to:[/bold green] {export_path}")


if __name__ == "__main__":
    main()
