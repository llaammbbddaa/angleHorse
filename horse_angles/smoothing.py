"""Bounded nearest-centre tracking for centred display smoothing."""
from .detect import LegDetection
import math


class Smoother:
    """Retain only the moving window and assign one-to-one nearest-centre IDs."""
    def __init__(self, window: int, width: int) -> None:
        self.window = window
        self.distance = .15 * width
        self.previous = []
        self.next_id = 0
        self.samples = []

    def add(self, index: int, detections: list[LegDetection]) -> list[int]:
        """Assign IDs and retain raw angle samples for this frame."""
        centres = [((d.box_xyxy[0] + d.box_xyxy[2]) / 2, (d.box_xyxy[1] + d.box_xyxy[3]) / 2) for d in detections]
        candidates = sorted((math.dist(c, p), i, identity) for i, c in enumerate(centres) for identity, p in self.previous)
        assigned, used = {}, set()
        for distance, i, identity in candidates:
            if distance <= self.distance and i not in assigned and identity not in used:
                assigned[i] = identity
                used.add(identity)
        for i in range(len(centres)):
            if i not in assigned:
                assigned[i] = self.next_id
                self.next_id += 1
        ids = [assigned[i] for i in range(len(centres))]
        self.previous = list(zip(ids, centres))
        self.samples.append((index, dict(zip(ids, [d.angle_deg for d in detections]))))
        self.samples = self.samples[-max(1, self.window):]
        return ids

    def values(self, index: int, ids: list[int], detections: list[LegDetection]) -> list[float | None]:
        """Average matched valid angles in the centred window, truncating boundaries."""
        left, right = (self.window - 1) // 2, self.window // 2
        result = []
        for identity, detection in zip(ids, detections):
            values = [angles[identity] for frame, angles in self.samples if index - left <= frame <= index + right and angles.get(identity) is not None]
            result.append(sum(values) / len(values) if values and detection.angle_deg is not None else None)
        return result
