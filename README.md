# Horse leg angle video annotator

A streaming Python 3.10+ CLI that applies the supplied Ultralytics pose model to a video and writes an annotated video plus full precision JSON and CSV results.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `ffmpeg` on PATH for H.264 output and original audio. Without it, the program keeps an mp4v video and prints a notice. For CPU-only installations, install CPU PyTorch wheels before the other requirements if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage

From this directory, with the virtual environment activated:

```bash
python main.py sample.mp4 --weights best.pt
python main.py sample.mp4 --weights best.zip --device cpu --max-frames 100
python main.py sample.mp4 --batch 8 --stride 2 --smooth 5 \
  --save-frames raw_frames --save-annotated-frames annotated_frames
pytest
```

`best.pt` is supplied in this workspace. ZIP files are searched recursively by member name for a name ending in `best.pt`; extraction is cached by archive content in `.model_cache/`. Only model bytes are extracted, so archive paths cannot escape the cache. When multiple matches exist, the first alphabetically is used.

| Argument | Default | Purpose |
|---|---|---|
| `INPUT_VIDEO` | required | Input mp4, mov, avi, mkv, webm, or another OpenCV-readable video |
| `--weights` | `best.pt` | Pose weights or ZIP containing best.pt |
| `--output` | `<input_stem>_annotated.mp4` | Destination video |
| `--conf` | `0.25` | Minimum box confidence |
| `--kp-conf` | `0.5` | Minimum confidence of each keypoint for an angle |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `mps`; auto selects CUDA, MPS, CPU |
| `--imgsz` | `640` | Inference image size |
| `--stride` | `1` | Infer every Nth frame; reuse detections on skipped frames |
| `--save-frames DIR` | off | Save raw frames, starting at frame_000001.jpg |
| `--save-annotated-frames DIR` | off | Save annotated JPEG frames |
| `--smooth N` | `0` | Centred display moving average; 0 disables |
| `--no-audio` | off | Omit source audio |
| `--max-frames N` | all | Limit output to the first N frames |
| `--batch` | `1` | Maximum contiguous chunk size; selected inference frames are batched |

## Outputs and conventions

The default outputs are `sample_annotated.mp4`, `sample_annotated_keypoints.json`, and `sample_annotated_angles.csv`. JSON contains one entry for every output frame, including empty detections; CSV contains one row per leg per frame. Frame indices start at zero and timestamps are `frame_index / source_fps`. Empty angles are JSON null / empty CSV fields. Raw angles are never rounded for export. A final summary reports counts, mean raw angle across detections, runtime, and output paths.

Keypoint order is **lower → middle → upper**, drawn red, yellow, white. The interior angle is measured at the middle point: a straight leg is 180 degrees. Degenerate segments shorter than 1e-6 pixels or any keypoint confidence below the threshold suppress the angle. Lines connect only confident endpoints. Blue boxes carry confidence labels, and valid angles have arcs and outlined text with a manually drawn degree circle. Positions come directly from YOLO in original image pixels.

Smoothing uses greedy one-to-one nearest box centres between consecutive output frames, limited to 15% of frame width. Unmatched legs start new histories; missing legs end their histories. The centred window includes `(N-1)//2` frames before and `N//2` after (even windows lean one frame forward), truncates at video boundaries, and ignores invalid angles. Invalid current angles remain blank. JSON and CSV retain raw angles and add `angle_smoothed_deg` when smoothing is enabled. Tracking identities are internal and are not anatomical identities. Left-to-right leg indices may change when legs cross.

Memory is bounded by the configured batch size and smoothing window, regardless of video length. Processing preserves decoded frame dimensions, FPS (30 if missing), and frame count unless limited explicitly. Input orientation follows OpenCV's decoder. Use an MP4 output for the documented codec/audio workflow. Audio is copied without transcoding, so an audio codec incompatible with the chosen output container produces a clear ffmpeg error, retaining the temporary mp4v file for recovery.

## Assumptions and deviations

The model is assumed to be a single-class YOLO pose model returning exactly three keypoints with confidences. Load failures report the original error and installed Ultralytics version. Output is one annotated video.

No user `keypoints.json` or sample video was present. Tests reconstruct the three supplied coordinate examples; their boxes/confidences are illustrative except for the supplied galloping example. Synthetic videos and mocked YOLO outputs exercise the pipeline without depending on real horse footage.

Empty-detection frames are left visually unchanged, including omission of the HUD, following the explicit pass-through requirement. Nonempty frames include frame/leg counts. H.264 yuv420p requires even dimensions; for odd dimensions, final encoding uses yuv444p to preserve resolution (OpenCV's mp4v writer may not support odd dimensions). The normal supported video path uses even dimensions. Skipped frames reuse the preceding raw detections and participate in smoothing and exports.

The second reference angle in the instructions is numerically inconsistent: its listed coordinates yield **178.42879415955596°**, not 178.2° (a difference of 0.2288°, beyond the requested ±0.2°). Tests use the mathematically computed value while preserving the specified formula and coordinates.
