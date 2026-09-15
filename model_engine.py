"""
Core ONNX Inference Engine for RT-DETR / YOLO Models
Supports image/video input, letterbox preprocessing, ONNX Runtime session execution,
postprocessing with coordinate scaling and optional NMS, and detailed metric tracking.
"""

import ast
from dataclasses import dataclass, field
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    """Individual object detection."""
    bbox: List[float]  # [x1, y1, x2, y2] in original image coordinates
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> Dict:
        return {
            "bbox": [round(float(coord), 2) for coord in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


@dataclass
class InferenceMetrics:
    """Detailed performance and latency metrics for an inference pass."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    detection_count: int = 0
    input_shape: Tuple[int, int] = (640, 640)
    device: str = "CPUExecutionProvider"

    def to_dict(self) -> Dict:
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "fps": round(self.fps, 1),
            "detection_count": self.detection_count,
            "input_shape": list(self.input_shape),
            "device": self.device,
        }


@dataclass
class InferenceResult:
    """Complete result of a single inference run."""
    original_image: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)


def compute_iou(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union (IoU) between box1 and an array of boxes."""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box1_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes."""
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class ONNXDetector:
    """
    High-performance detector running ONNX models with ONNX Runtime.
    Handles RT-DETR and standard YOLO architectures seamlessly.
    """

    def __init__(
        self,
        model_path: str = "best.onnx",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "auto",
        custom_classes: Optional[Union[Dict, List, str]] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at path: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Determine execution providers
        available_providers = ort.get_available_providers()
        self.providers = self._select_providers(device, available_providers)

        # Initialize ONNX session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=self.providers,
        )

        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Parse model metadata and class names
        self.names = self._parse_metadata(custom_classes)

    def _select_providers(self, device: str, available: List[str]) -> List[str]:
        device = device.lower()
        if device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml" and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            return ["CPUExecutionProvider"]
        else:
            # Auto selection
            for pref in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]:
                if pref in available:
                    return [pref, "CPUExecutionProvider"] if pref != "CPUExecutionProvider" else ["CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def _format_class_dict(self, classes: Union[Dict, List, str]) -> Dict[int, str]:
        """Convert various class input representations into {int_id: class_name}."""
        if isinstance(classes, dict):
            return {int(k): str(v) for k, v in classes.items()}
        elif isinstance(classes, (list, tuple)):
            return {i: str(v) for i, v in enumerate(classes)}
        elif isinstance(classes, str):
            # If path to a file
            if os.path.isfile(classes):
                with open(classes, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    return {i: line for i, line in enumerate(lines)}
            # Comma-separated list
            parts = [p.strip() for p in classes.split(",") if p.strip()]
            return {i: p for i, p in enumerate(parts)}
        return {0: "object"}

    def _parse_metadata(self, custom_classes: Optional[Union[Dict, List, str]] = None) -> Dict[int, str]:
        """Extract class names and configuration from ONNX model metadata or custom override."""
        if custom_classes:
            return self._format_class_dict(custom_classes)

        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                raw_names = meta["names"]
                # 1. Try ast.literal_eval (standard Python dict representation saved by Ultralytics: {0: 'helmet'})
                try:
                    parsed = ast.literal_eval(raw_names)
                    if isinstance(parsed, (dict, list)):
                        return self._format_class_dict(parsed)
                except Exception:
                    pass

                # 2. Try json.loads
                try:
                    parsed = json.loads(raw_names)
                    if isinstance(parsed, (dict, list)):
                        return self._format_class_dict(parsed)
                except Exception:
                    pass

                # 3. Try yaml safe_load
                try:
                    import yaml
                    parsed = yaml.safe_load(raw_names)
                    if isinstance(parsed, (dict, list)):
                        return self._format_class_dict(parsed)
                except Exception:
                    pass
        except Exception:
            pass

        return {0: "object"}

    def set_thresholds(self, conf: Optional[float] = None, iou: Optional[float] = None):
        """Update confidence and IoU thresholds dynamically."""
        if conf is not None:
            self.conf_threshold = max(0.01, min(1.0, conf))
        if iou is not None:
            self.iou_threshold = max(0.01, min(1.0, iou))

    def letterbox(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with preserved aspect ratio and pad to target_size (height, width).
        Returns: (padded_image, scale_factor, (pad_w, pad_h))
        """
        ih, iw = image.shape[:2]
        th, tw = target_size

        scale = min(tw / iw, th / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))

        if (iw, ih) != (nw, nh):
            resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        pad_w = (tw - nw) / 2
        pad_h = (th - nh) / 2

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, scale, (left, top)

    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess input BGR image for ONNX model:
        1. Letterbox resize with padding
        2. BGR to RGB conversion
        3. Normalization to [0.0, 1.0]
        4. HWC -> CHW -> NCHW transpose
        """
        padded_img, scale, pad = self.letterbox(image, self.input_size)
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        norm_img = rgb_img.astype(np.float32) / 255.0
        chw_img = np.transpose(norm_img, (2, 0, 1))
        nchw_img = np.expand_dims(chw_img, axis=0)
        return np.ascontiguousarray(nchw_img, dtype=np.float32), scale, pad

    def postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[Detection]:
        """
        Parse raw model predictions into scaled Bounding Boxes and Detections.
        Handles RT-DETR decoder output format: shape [1, num_queries, 6]
        where the 6 values are [cx, cy, w, h, score, class_id] in normalized [0, 1] coords.
        """
        preds = raw_output[0]  # shape: (num_queries, 6) or (num_queries, 4 + num_classes)
        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape
        detections: List[Detection] = []

        if preds.ndim != 2:
            return detections

        # Check RT-DETR format: (num_queries, 6) where columns are [cx, cy, w, h, score, class_id]
        if preds.shape[1] == 6:
            cx = preds[:, 0]
            cy = preds[:, 1]
            w = preds[:, 2]
            h = preds[:, 3]
            scores = preds[:, 4]
            class_ids = preds[:, 5].astype(int)

            # Convert normalized cx, cy, w, h to letterboxed pixel coordinates
            th, tw = self.input_size
            x1 = (cx - w / 2.0) * tw
            y1 = (cy - h / 2.0) * th
            x2 = (cx + w / 2.0) * tw
            y2 = (cy + h / 2.0) * th

            # Remove padding and rescale back to original image dimensions
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clip to image boundaries
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)

            # Filter by confidence threshold
            conf_mask = scores >= self.conf_threshold
            if not np.any(conf_mask):
                return []

            filtered_boxes = np.column_stack([x1[conf_mask], y1[conf_mask], x2[conf_mask], y2[conf_mask]])
            filtered_scores = scores[conf_mask]
            filtered_classes = class_ids[conf_mask]

            # Apply NMS if multiple detections exist
            if len(filtered_boxes) > 1 and self.iou_threshold < 1.0:
                keep_indices = nms(filtered_boxes, filtered_scores, self.iou_threshold)
            else:
                keep_indices = list(range(len(filtered_boxes)))

            for idx in keep_indices:
                box = filtered_boxes[idx].tolist()
                score = float(filtered_scores[idx])
                cid = int(filtered_classes[idx])
                cname = self.names.get(cid, f"class_{cid}")
                detections.append(
                    Detection(
                        bbox=box,
                        confidence=score,
                        class_id=cid,
                        class_name=cname,
                    )
                )

        return detections

    def predict(self, image: np.ndarray) -> InferenceResult:
        """
        Run inference on a single BGR image array with high-precision latency measurement.
        """
        t0 = time.perf_counter()

        # 1. Preprocess
        input_tensor, scale, pad = self.preprocess(image)
        t1 = time.perf_counter()

        # 2. Run ONNX session
        raw_outputs = self.session.run(
            [self.output_name], {self.input_name: input_tensor}
        )
        t2 = time.perf_counter()

        # 3. Postprocess
        detections = self.postprocess(
            raw_outputs[0], image.shape[:2], scale, pad
        )
        t3 = time.perf_counter()

        # Latency calculations (milliseconds)
        preprocess_ms = (t1 - t0) * 1000.0
        inference_ms = (t2 - t1) * 1000.0
        postprocess_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        metrics = InferenceMetrics(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            fps=fps,
            detection_count=len(detections),
            input_shape=self.input_size,
            device=self.active_provider,
        )

        return InferenceResult(
            original_image=image,
            detections=detections,
            metrics=metrics,
        )
