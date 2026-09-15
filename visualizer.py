"""
High-Performance Visualizer for Object Detection and Inference Metrics
Renders aesthetic bounding boxes, label tags, and a Heads-Up Display (HUD) overlay
displaying real-time latency breakdown (preprocess, inference, postprocess), FPS, and detection counts.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from model_engine import Detection, InferenceMetrics


# Curated color palette (BGR format for OpenCV)
CLASS_COLORS = [
    (246, 178, 64),   # Cyan/Sky blue
    (76, 175, 80),    # Emerald green
    (38, 149, 243),   # Electric blue
    (156, 39, 176),   # Purple
    (233, 30, 99),    # Pink
    (255, 152, 0),    # Amber
    (0, 188, 212),    # Teal
]


class Visualizer:
    """
    Renders detections and inference performance metrics onto frames.
    """

    def __init__(
        self,
        show_hud: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        hud_position: str = "top-left",
    ):
        self.show_hud = show_hud
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.hud_position = hud_position

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        return CLASS_COLORS[class_id % len(CLASS_COLORS)]

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draw bounding boxes and class badges on the image."""
        annotated = image.copy()
        img_h, img_w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = [int(round(c)) for c in det.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

            color = self._get_color(det.class_id)

            # Draw main bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                thickness=self.box_thickness,
                lineType=cv2.LINE_AA,
            )

            # Draw label tag
            if self.show_labels:
                label = det.class_name
                if self.show_conf:
                    label += f" {det.confidence * 100:.1f}%"

                (tw, th), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, thickness=1
                )
                
                # Position label above box if space permits, else inside
                tag_y1 = max(0, y1 - th - baseline - 6)
                tag_y2 = y1
                if y1 - th - baseline - 6 < 0:
                    tag_y1 = y1
                    tag_y2 = y1 + th + baseline + 6

                tag_x2 = min(img_w, x1 + tw + 10)

                # Draw label background badge
                cv2.rectangle(
                    annotated,
                    (x1, tag_y1),
                    (tag_x2, tag_y2),
                    color,
                    cv2.FILLED,
                )

                # Determine text contrast color (dark or white)
                # Compute luminance
                b, g, r = color
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, tag_y2 - baseline - 3),
                    self.font,
                    self.font_scale,
                    text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        return annotated

    def draw_hud(
        self,
        image: np.ndarray,
        metrics: InferenceMetrics,
        model_name: str = "best.onnx",
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        extra_info: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw a modern translucent Heads-Up Display (HUD) card showing metrics.
        """
        if not self.show_hud:
            return image

        annotated = image.copy()
        img_h, img_w = annotated.shape[:2]

        lines = [
            f"Model: {model_name} ({metrics.device})",
            f"FPS: {metrics.fps:.1f}  |  Total: {metrics.total_ms:.1f}ms",
            f"Pre: {metrics.preprocess_ms:.1f}ms | Infer: {metrics.inference_ms:.1f}ms | Post: {metrics.postprocess_ms:.1f}ms",
            f"Detections: {metrics.detection_count}"
            + (f" (Conf >= {conf_thresh:.2f})" if conf_thresh is not None else ""),
        ]
        if extra_info:
            lines.append(extra_info)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        line_height = 20
        padding = 10

        # Calculate HUD dimensions
        max_width = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, 1)
            if w > max_width:
                max_width = w

        hud_w = max_width + padding * 2
        hud_h = len(lines) * line_height + padding * 2

        # Position HUD
        if self.hud_position == "top-right":
            x1 = img_w - hud_w - 15
            y1 = 15
        else:  # top-left default
            x1 = 15
            y1 = 15

        x2 = x1 + hud_w
        y2 = y1 + hud_h

        # Make sure within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        # Draw semi-transparent background overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 33), cv2.FILLED)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 85, 105), 1, cv2.LINE_AA)

        # Alpha blend HUD background
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Draw text lines
        text_y = y1 + padding + 14
        for idx, line in enumerate(lines):
            # Highlight FPS / Detections
            if "FPS" in line:
                color = (76, 217, 100)  # Bright green
            elif "Detections:" in line and metrics.detection_count > 0:
                color = (0, 215, 255)  # Gold/Yellow
            elif idx == 0:
                color = (255, 255, 255)  # Crisp white
            else:
                color = (195, 205, 220)  # Light gray-blue

            cv2.putText(
                annotated,
                line,
                (x1 + padding, text_y),
                font,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            text_y += line_height

        return annotated

    def render(
        self,
        image: np.ndarray,
        detections: List[Detection],
        metrics: InferenceMetrics,
        model_name: str = "best.onnx",
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        extra_info: Optional[str] = None,
    ) -> np.ndarray:
        """Convenience method to draw both detections and HUD overlay."""
        annotated = self.draw_detections(image, detections)
        if self.show_hud:
            annotated = self.draw_hud(
                annotated,
                metrics,
                model_name=model_name,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                extra_info=extra_info,
            )
        return annotated
