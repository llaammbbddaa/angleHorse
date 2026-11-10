"""Incremental JSON and CSV export."""
from .detect import LegDetection
import csv
import json
from pathlib import Path


class ResultWriter:
    """Stream aligned results without retaining frames or detections."""
    def __init__(self, output: Path, smooth: bool = False) -> None:
        self.json_path = output.with_name(output.stem + '_keypoints.json')
        self.csv_path = output.with_name(output.stem + '_angles.csv')
        self.json = self.json_path.open('w')
        self.csv_file = self.csv_path.open('w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.smooth = smooth
        self.first = True
        self.json.write('[\n')
        self.csv.writerow(['frame_index', 'timestamp_sec', 'leg_index', 'box_conf', 'angle_deg'] + [f'kp{i}_{s}' for i in range(3) for s in ('x', 'y', 'conf')] + (['angle_smoothed_deg'] if smooth else []))

    def write(self, index: int, fps: float, detections: list[LegDetection], smoothed: list[float | None]) -> None:
        """Write one frame and one CSV row per leg, preserving raw precision."""
        entry = dict(frame_index=index, timestamp_sec=index / fps,
                     boxes_xyxy=[d.box_xyxy for d in detections], confidence=[d.box_conf for d in detections],
                     keypoints_xy_conf=[d.keypoints for d in detections], angles_deg=[d.angle_deg for d in detections])
        if self.smooth:
            entry['angle_smoothed_deg'] = smoothed
        if not self.first:
            self.json.write(',\n')
        json.dump(entry, self.json, allow_nan=False)
        self.first = False
        for i, d in enumerate(detections):
            self.csv.writerow([index, index / fps, i, d.box_conf, d.angle_deg] + [v for p in d.keypoints for v in p] + ([smoothed[i]] if self.smooth else []))

    def close(self) -> None:
        """Finish JSON and close both files."""
        self.json.write('\n]\n')
        self.json.close()
        self.csv_file.close()
