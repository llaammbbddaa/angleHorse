"""Command-line horse leg video annotator."""
from typing import Any
import argparse
from collections import deque
from pathlib import Path
import tempfile
import time
import sys
import cv2
from tqdm import tqdm
from horse_angles.model import load_model, select_device
from horse_angles.detect import detect_batch
from horse_angles.draw import annotate_frame
from horse_angles.export import ResultWriter
from horse_angles.smoothing import Smoother
from horse_angles.video_io import FrameReader, finalize_video


def build_parser() -> argparse.ArgumentParser:
    """Build the documented CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_video')
    parser.add_argument('--weights', default='best.pt')
    parser.add_argument('--output')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--kp-conf', type=float, default=.5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--save-frames')
    parser.add_argument('--save-annotated-frames')
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--batch', type=int, default=1)
    return parser


def run_pipeline(args: argparse.Namespace, model: Any = None) -> dict:
    """Stream inference, annotation and exports with bounded batching/lookahead."""
    start = time.monotonic()
    if min(args.stride, args.batch, args.imgsz) < 1 or args.smooth < 0 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError('stride, batch, imgsz and max-frames must be positive; smooth must be nonnegative')
    if not 0 <= args.conf <= 1 or not 0 <= args.kp_conf <= 1:
        raise ValueError('Confidence thresholds must lie between 0 and 1')
    source = Path(args.input_video)
    output = Path(args.output) if args.output else source.with_name(source.stem + '_annotated.mp4')
    if source.resolve() == output.resolve():
        raise ValueError('Output must differ from input')
    if not source.is_file():
        raise ValueError(f'Missing input file: {source}')
    if model is None:
        model = load_model(args.weights)
        device = select_device(args.device)
    else:
        device = 'cpu' if args.device == 'auto' else args.device
    reader = FrameReader(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='.horse_angles_', dir=output.parent)
    import os
    os.close(handle)
    temporary = Path(temp_path)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), reader.fps, (reader.width, reader.height))
    if not writer.isOpened():
        reader.capture.release()
        raise ValueError('Could not open output video writer')
    results = ResultWriter(output, bool(args.smooth))
    total = reader.total_frames
    if args.max_frames is not None:
        total = min(total, args.max_frames) if total else args.max_frames
    directory = Path(args.save_annotated_frames) if args.save_annotated_frames else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    smoother = Smoother(args.smooth, reader.width)
    pending = deque()
    previous = []
    count = detected_count = angle_count = 0
    angle_sum = 0.0
    progress = tqdm(total=total, unit='frame')

    def emit():
        nonlocal count, detected_count, angle_count, angle_sum
        index, frame, detections, identities = pending.popleft()
        smoothed = smoother.values(index, identities, detections) if args.smooth else [d.angle_deg for d in detections]
        annotated = annotate_frame(frame, detections, index, total, args.kp_conf, smoothed)
        writer.write(annotated)
        if directory and not cv2.imwrite(str(directory / f'frame_{index + 1:06d}.jpg'), annotated):
            raise ValueError('Could not save annotated frame')
        results.write(index, reader.fps, detections, smoothed)
        count += 1
        detected_count += bool(detections)
        for detection in detections:
            if detection.angle_deg is not None:
                angle_sum += detection.angle_deg
                angle_count += 1
        progress.update(1)

    # A contiguous chunk bounds memory even when stride is very large.
    frames = reader.frames(args.max_frames, args.save_frames)
    try:
        while True:
            chunk = []
            for _ in range(args.batch):
                try:
                    chunk.append(next(frames))
                except StopIteration:
                    break
            if not chunk:
                break
            selected = [frame for index, frame in chunk if index % args.stride == 0]
            predictions = iter(detect_batch(model, selected, args.conf, args.kp_conf, args.imgsz, device) if selected else [])
            for index, frame in chunk:
                if index % args.stride == 0:
                    previous = next(predictions)
                identities = smoother.add(index, previous) if args.smooth else []
                pending.append((index, frame, previous, identities))
                if len(pending) > args.smooth // 2:
                    emit()
        while pending:
            emit()
    finally:
        frames.close()
        reader.capture.release()
        writer.release()
        results.close()
        progress.close()
    finalize_video(temporary, output, str(source), args.no_audio)
    summary = dict(frames=count, frames_with_detections=detected_count,
                   mean_angle=angle_sum / angle_count if angle_count else None,
                   output=str(output), json=str(results.json_path), csv=str(results.csv_path),
                   runtime_sec=time.monotonic() - start)
    mean = f'{summary["mean_angle"]:.3f}' if angle_count else 'n/a'
    print(f'Frames: {count}; with detections: {detected_count}; mean angle: {mean}; runtime: {summary["runtime_sec"]:.2f}s')
    print(f'Video: {output}\nJSON: {results.json_path}\nCSV: {results.csv_path}')
    return summary


def main() -> int:
    """Run the CLI, reporting actionable failures without tracebacks."""
    try:
        run_pipeline(build_parser().parse_args())
        return 0
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
