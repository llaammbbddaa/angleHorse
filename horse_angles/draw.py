"""Scaled OpenCV overlays."""
from typing import Any
import math
import cv2
from .detect import LegDetection


def annotate_frame(frame: Any, detections: list[LegDetection], index: int, total: int | None,
                   kp_conf: float = .5, displayed: list[float | None] | None = None) -> Any:
    """Draw leg geometry and a HUD on a copy; empty frames pass through unchanged."""
    output = frame.copy()
    if not detections:
        return output
    scale = max(frame.shape[:2]) / 1000
    thickness = max(1, round(2 * scale))
    radius = max(2, round(5 * scale))
    font = max(.3, .65 * scale)
    def text(value, position, color=(255, 255, 255)):
        cv2.putText(output, value, position, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(output, value, position, cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
    for i, leg in enumerate(detections):
        x1, y1, x2, y2 = map(round, leg.box_xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        label = f'leg {leg.box_conf:.2f}'
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)
        tag_y = max(h + baseline + 2, y1)
        cv2.rectangle(output, (x1, tag_y - h - baseline - 2), (x1 + w + 4, tag_y), (255, 0, 0), -1)
        cv2.putText(output, label, (x1 + 2, tag_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thickness, cv2.LINE_AA)
        points = [tuple(map(round, p[:2])) for p in leg.keypoints]
        valid = [p[2] >= kp_conf for p in leg.keypoints]
        for a, b in ((0, 1), (1, 2)):
            if valid[a] and valid[b]:
                cv2.line(output, points[a], points[b], (0, 255, 0), thickness, cv2.LINE_AA)
        for point, good, color in zip(points, valid, ((0, 0, 255), (0, 255, 255), (255, 255, 255))):
            if good:
                cv2.circle(output, point, radius + thickness, (25, 25, 25), -1, cv2.LINE_AA)
                cv2.circle(output, point, radius, color, -1, cv2.LINE_AA)
        angle = displayed[i] if displayed is not None else leg.angle_deg
        if angle is not None and leg.angle_deg is not None:
            a, b, c = leg.keypoints
            start = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))
            end = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
            delta = (end - start + 180) % 360 - 180
            r = max(20 * scale, min(60 * scale, .12 * min(math.dist(a[:2], b[:2]), math.dist(c[:2], b[:2]))))
            cv2.ellipse(output, points[1], (max(1, round(r)),) * 2, 0, min(start, start + delta), max(start, start + delta), (255, 255, 255), thickness, cv2.LINE_AA)
            value = f'{angle:.1f}'
            position = (max(0, min(frame.shape[1] - round(100 * scale), points[1][0] + round(20 * scale))), max(15, points[1][1] - round(15 * scale)))
            text(value, position)
            (w, h), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)
            center = (position[0] + w + radius, position[1] - h)
            cv2.circle(output, center, max(1, radius // 2), (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.circle(output, center, max(1, radius // 2), (255, 255, 255), thickness, cv2.LINE_AA)
    text(f'Frame {index + 1} / {total or "?"}   Legs: {len(detections)}', (10, max(20, round(30 * scale))))
    return output
