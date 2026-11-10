"""Convert YOLO pose results into leg detections."""
from dataclasses import dataclass
from typing import Any
from .geometry import compute_angle


@dataclass
class LegDetection:
    """One leg, ordered lower, middle, upper, in original image pixels."""
    box_xyxy: tuple[float, float, float, float]
    box_conf: float
    keypoints: list[tuple[float, float, float]]
    angle_deg: float | None


def parse_arrays(boxes: Any, confidences: Any, keypoints: Any, kp_conf: float = 0.5) -> list[LegDetection]:
    """Parse aligned box and three-keypoint arrays, sorted by horizontal centre."""
    if not (len(boxes) == len(confidences) == len(keypoints)):
        raise ValueError('Detection arrays have different lengths')
    detections = []
    for box, confidence, points in zip(boxes, confidences, keypoints):
        if len(box) != 4 or len(points) != 3 or any(len(p) != 3 for p in points):
            raise ValueError('Expected four box coordinates and exactly three (x, y, confidence) keypoints')
        points = [tuple(map(float, p)) for p in points]
        angle = compute_angle(*points) if all(p[2] >= kp_conf for p in points) else None
        detections.append(LegDetection(tuple(map(float, box)), float(confidence), points, angle))
    return sorted(detections, key=lambda d: d.box_xyxy[0] + d.box_xyxy[2])


def detect_batch(model: Any, frames: list, conf: float, kp_conf: float, imgsz: int, device: str) -> list[list[LegDetection]]:
    """Run a batch and preserve frame order."""
    results = list(model.predict(frames, conf=conf, imgsz=imgsz, device=device, verbose=False))
    if len(results) != len(frames):
        raise ValueError('Model returned an incorrect number of frame results')
    output = []
    for result in results:
        if result.keypoints is None or result.boxes is None or len(result.boxes.xyxy) == 0:
            output.append([])
            continue
        import numpy as np
        xy = result.keypoints.xy.cpu().numpy()
        confidence = result.keypoints.conf
        if confidence is None:
            raise ValueError('Pose model must return per-keypoint confidence')
        points = np.concatenate((xy, confidence.cpu().numpy()[..., None]), axis=-1)
        output.append(parse_arrays(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), points, kp_conf))
    return output
