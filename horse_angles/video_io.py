"""Streaming video input and output finalization."""
from typing import Any, Iterator
from pathlib import Path
import math
import shutil
import subprocess
import cv2


class FrameReader:
    """Own a capture and expose actual decoded dimensions."""
    def __init__(self, path: str) -> None:
        if not Path(path).is_file():
            raise ValueError(f'Missing input file: {path}')
        self.capture = cv2.VideoCapture(str(path))
        ok, self.first = self.capture.read()
        if not ok:
            self.capture.release()
            raise ValueError(f'Unreadable video: {path}')
        self.height, self.width = self.first.shape[:2]
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(total) if math.isfinite(total) and total > 0 else None

    def frames(self, maximum: int | None = None, save_dir: str | None = None) -> Iterator[tuple[int, Any]]:
        """Yield indexed frames, saving raw frames on request."""
        directory = Path(save_dir) if save_dir else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        index, frame = 0, self.first
        try:
            while maximum is None or index < maximum:
                if directory and not cv2.imwrite(str(directory / f'frame_{index + 1:06d}.jpg'), frame):
                    raise ValueError('Could not save raw frame')
                yield index, frame
                index += 1
                if maximum is not None and index >= maximum:
                    break
                ok, frame = self.capture.read()
                if not ok:
                    break
        finally:
            self.capture.release()


def finalize_video(temporary: Path, output: Path, source: str, no_audio: bool) -> None:
    """Encode H.264 and optionally copy source audio, or retain mp4v."""
    executable = shutil.which('ffmpeg')
    if not executable:
        print('ffmpeg unavailable; keeping mp4v video without audio.')
        temporary.replace(output)
        return
    command = [executable, '-y', '-v', 'error', '-i', str(temporary)]
    if not no_audio:
        command += ['-i', str(source)]
    command += ['-map', '0:v']
    if not no_audio:
        command += ['-map', '1:a?', '-c:a', 'copy']
    capture = cv2.VideoCapture(str(temporary))
    try:
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    command += ['-t', str(duration)]
    command += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(output)]
    # Preserve odd dimensions by using yuv444p rather than padding pixels.
    if any(v % 2 for v in _dimensions(temporary)):
        command[command.index('yuv420p')] = 'yuv444p'
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'ffmpeg failed: {result.stderr.strip()} (temporary video retained at {temporary})')
    temporary.unlink()


def _dimensions(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(3)), int(capture.get(4))
    finally:
        capture.release()
