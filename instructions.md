# Horse Leg Angle Video Annotator: Build Instructions

You are building a Python command-line program that takes a video of a horse, runs a custom-trained leg keypoint model on every frame, calculates the joint angle of each detected leg, and writes out a new video with the keypoints and angles drawn on it.

Read this whole document before writing code. Build exactly what is described. Where something is marked **configurable**, expose it as a CLI flag with the stated default.

---

## 1. Goal

```
input video  ->  frames  ->  best.pt (leg keypoints)  ->  angle per leg  ->  annotated output video
```

1. Slice the input video into frames.
2. Run each frame through the pretrained `best.pt` model to get leg detections and 3 keypoints per leg.
3. Compute the angle formed by the 3 keypoints.
4. Draw the box, keypoints, connecting lines, and angle on each frame.
5. Reassemble the frames into an output video (same resolution and FPS as the input).
6. Also save a machine-readable results file (JSON + CSV) with every detection and angle.

---

## 2. What the model outputs (ground truth from the provided examples)

`best.pt` is an Ultralytics YOLO **pose** model with a single class, `leg`. For each detected leg it returns:

- a bounding box (`xyxy`, absolute pixel coordinates)
- a box confidence
- **exactly 3 keypoints**, each `(x, y, confidence)`, in absolute pixel coordinates of the image that was passed in

A sample of the output format, `keypoints.json`, looks like this (one entry per image; one inner list per detected leg):

```json
{
  "image": "path/to/image.jpg",
  "boxes_xyxy": [[917.7, 600.3, 1142.4, 1064.7]],
  "confidence": [0.9595],
  "keypoints_xy_conf": [
    [
      [933.5, 997.5, 0.972],
      [1076.7, 896.1, 0.986],
      [1112.3, 678.6, 0.996]
    ]
  ]
}
```

### Keypoint order (important)

Looking at the annotated sample images, the three keypoints are always ordered **bottom to top** along the leg:

| Index | Dot colour in the sample images | Location on leg |
|-------|----------------------------------|-----------------|
| 0     | **red**                          | lowest point (fetlock/hoof end) |
| 1     | **yellow**                       | middle joint (knee on a foreleg, hock on a hind leg) |
| 2     | **white**                        | highest point (upper leg: shoulder/elbow area on a foreleg, hip/stifle area on a hind leg) |

Use these colours when drawing so the output matches the images the model was validated on. Do not hard-code anatomical names beyond "lower / middle / upper"; use those as the labels.

---

## 3. Angle calculation

The angle is the **interior angle at the middle keypoint (index 1)** between the segment to the lower point (index 0) and the segment to the upper point (index 2).

```
A = keypoint[0]   (lower)
B = keypoint[1]   (middle, the vertex)
C = keypoint[2]   (upper)

v1 = A - B
v2 = C - B

angle = degrees( arccos( dot(v1, v2) / (|v1| * |v2|) ) )
```

Rules:

- Clamp the cosine to `[-1, 1]` before `arccos` to avoid NaNs from floating-point error.
- If either vector has length 0 (or < 1e-6), the angle is `None`.
- Result is in degrees, range 0 to 180. A perfectly straight leg is about 180°; a fully flexed joint approaches 0°.
- Round to 1 decimal place **only for display**; store full precision in the JSON/CSV.
- Implement as a standalone, pure function: `compute_angle(a, b, c) -> float | None`.

### Reference values for unit tests

Use the three real model outputs from `keypoints.json` to write tests (tolerance ±0.2°):

| Image | Lower (kp0) | Middle (kp1) | Upper (kp2) | Expected angle |
|-------|-------------|--------------|-------------|----------------|
| `2023-09-ClippingV3-sq` | (483.94, 500.70) | (493.82, 414.69) | (482.45, 293.65) | ≈ **167.9°** |
| `4-1024x683` | (494.51, 579.88) | (524.24, 506.43) | (563.72, 400.64) | ≈ **178.2°** |
| `lh-pinto-horse...galloping` | (933.53, 997.55) | (1076.73, 896.14) | (1112.38, 678.67) | ≈ **134.6°** |

Also test: three collinear points -> 180°, a right angle -> 90°, and duplicate points -> `None`.

---

## 4. Tech stack and environment

- Python 3.10+
- `ultralytics` (loads `best.pt` via `YOLO("best.pt")`)
- `opencv-python` (video read/write, drawing)
- `numpy`
- `tqdm` (progress bar)
- `pytest` (tests)
- Optional but recommended: `ffmpeg` on PATH for H.264 re-encoding and copying the original audio

Provide a `requirements.txt` and a short `README.md` with setup and usage.

**Model file:** the pretrained weights were provided as a **zip file** containing `best.pt`. The program must accept either:

- a path to `best.pt`, or
- a path to the `.zip`, in which case it extracts `best.pt` to a temp/cache folder (`.model_cache/`) on first use and reuses it afterward. Search the zip for any file ending in `best.pt`, since it may be inside a subfolder.

Use the GPU automatically if available (`device` **configurable**, default `auto`: CUDA, then Apple MPS, then CPU).

---

## 5. Project layout

```
horse_leg_angles/
├── requirements.txt
├── README.md
├── main.py                 # CLI entry point
├── horse_angles/
│   ├── __init__.py
│   ├── model.py            # load_model(), resolve weights path (zip or .pt)
│   ├── video_io.py         # frame reader generator, video writer, ffmpeg helpers
│   ├── geometry.py         # compute_angle()
│   ├── detect.py           # run model on a frame -> list[LegDetection]
│   ├── draw.py             # annotate_frame()
│   └── export.py           # JSON / CSV writers
└── tests/
    ├── test_geometry.py
    ├── test_detect_schema.py
    └── test_pipeline_smoke.py
```

### Core data structure

```python
@dataclass
class LegDetection:
    box_xyxy: tuple[float, float, float, float]
    box_conf: float
    keypoints: list[tuple[float, float, float]]   # 3 x (x, y, conf), order: lower, middle, upper
    angle_deg: float | None                       # None if not computable / low confidence
```

---

## 6. CLI

```
python main.py INPUT_VIDEO --weights best.pt [options]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `INPUT_VIDEO` | required | Path to input video (mp4, mov, avi, mkv, webm) |
| `--weights` | `best.pt` | Path to `best.pt` **or** the zip containing it |
| `--output` | `<input_stem>_annotated.mp4` | Output video path |
| `--conf` | `0.25` | Minimum **box** confidence passed to the model |
| `--kp-conf` | `0.5` | Minimum **per-keypoint** confidence required to compute/draw an angle |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `--imgsz` | `640` | Model inference size |
| `--stride` | `1` | Process every Nth frame (skipped frames reuse the previous detections in the output so the video stays full length) |
| `--save-frames DIR` | off | Also write each extracted raw frame as `frame_000001.jpg` into DIR |
| `--save-annotated-frames DIR` | off | Also write each annotated frame into DIR |
| `--smooth N` | `0` | Moving-average window for displayed angles (0 = off), see §9 |
| `--no-audio` | off | Skip copying audio from the source |
| `--max-frames N` | none | Process only the first N frames (for quick tests) |
| `--batch` | `1` | Frames per inference batch (raise on GPU for speed) |

Exit with a clear error message (not a stack trace) for: missing input file, unreadable video, missing weights, zip without `best.pt`.

---

## 7. Pipeline details

### 7.1 Frame slicing (`video_io.py`)

- Use `cv2.VideoCapture`. Read **frame by frame as a generator**; never load the whole video into memory.
- Read and keep: `fps` (fall back to 30.0 if the container reports 0), `width`, `height`, `total_frames` (may be approximate; the progress bar should tolerate this).
- Handle rotated phone videos: OpenCV normally auto-applies orientation, but verify that output width/height match the frames actually read (use the first frame's shape, not the container metadata).
- If `--save-frames` is set, write each raw frame to disk as it is read.

### 7.2 Detection (`detect.py`)

```python
results = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
r = results[0]
boxes   = r.boxes.xyxy.cpu().numpy()      # (N, 4)
confs   = r.boxes.conf.cpu().numpy()      # (N,)
kp_xy   = r.keypoints.xy.cpu().numpy()    # (N, 3, 2)
kp_conf = r.keypoints.conf.cpu().numpy()  # (N, 3)
```

- Coordinates are already in original-frame pixels; do not rescale.
- Handle `r.keypoints is None` or zero detections (return an empty list; the frame is still written to the output unmodified).
- A frame may contain **multiple legs** (a horse has up to four, and the sample galloping image shows a second horse). Process all detections. Sort them left-to-right by box centre x so the on-screen order is stable.
- Compute the angle only if all three keypoint confidences are `>= --kp-conf`; otherwise `angle_deg = None`.
- When `--batch > 1`, pass a list of frames to `model.predict` and map results back to frame indices in order.

### 7.3 Annotation (`draw.py`)

For each `LegDetection` draw, in this order:

1. **Bounding box**: blue (`(255, 0, 0)` in BGR), label `leg 0.85` (class name + box confidence to 2 decimals) on a filled blue tag above the box. This mirrors the sample images.
2. **Limb lines**: lines connecting kp0 -> kp1 -> kp2 in a contrasting colour (e.g. green).
3. **Keypoints**: filled circles, **kp0 red, kp1 yellow, kp2 white**, with a thin dark outline so they show on any background. Skip any keypoint below `--kp-conf`.
4. **Angle arc** at kp1: draw a small arc (radius about 12% of the shorter segment, clamped to 20–60 px) between the two segments to visualise the angle.
5. **Angle text**: `168.0°` placed near the middle keypoint, offset so it doesn't sit on the limb lines. Use white text with a black outline (draw text twice: thick black first, thin white on top) for readability. If the angle is `None`, draw nothing for the text.

Scaling: all line thicknesses, circle radii, and font scales must scale with frame size (e.g. `scale = max(width, height) / 1000`), so annotations look right on 480p and 4K alike.

Unicode note: `cv2.putText` cannot render `°` in Hershey fonts. Either draw a small circle manually after the number, or use the `°` via Pillow text rendering. Pick the manual-circle approach to avoid the extra dependency.

Also add a small HUD in the top-left corner: `Frame 123 / 900` and the number of legs detected.

### 7.4 Writing the output video

- Write annotated frames with `cv2.VideoWriter` (`mp4v` fourcc) to a **temporary** file at the source FPS and resolution.
- After all frames are written, if `ffmpeg` is available:
  - re-encode to H.264 (`-c:v libx264 -pix_fmt yuv420p -crf 18`) for broad compatibility, and
  - copy the original audio track if one exists (unless `--no-audio`): `-map 0:v -map 1:a? -c:a copy`.
- If `ffmpeg` is not available, keep the `mp4v` file as the final output and print a one-line notice.
- Delete temp files on success.

---

## 8. Result exports (`export.py`)

Write next to the output video, using the same stem:

### `<stem>_keypoints.json`

Same schema as the provided `keypoints.json`, **one entry per frame**, plus extra fields:

```json
[
  {
    "frame_index": 0,
    "timestamp_sec": 0.0,
    "boxes_xyxy": [[917.7, 600.3, 1142.4, 1064.7]],
    "confidence": [0.9595],
    "keypoints_xy_conf": [[[933.5, 997.5, 0.972], [1076.7, 896.1, 0.986], [1112.3, 678.6, 0.996]]],
    "angles_deg": [134.6]
  }
]
```

- `angles_deg` is index-aligned with `boxes_xyxy`; use `null` where an angle was not computed.
- Frames with no detections still get an entry with empty lists.
- Stream the JSON to disk (or write at the end, but without holding annotated images in memory).

### `<stem>_angles.csv`

One row per **leg per frame**:

```
frame_index,timestamp_sec,leg_index,box_conf,angle_deg,kp0_x,kp0_y,kp0_conf,kp1_x,kp1_y,kp1_conf,kp2_x,kp2_y,kp2_conf
```

---

## 9. Optional smoothing (`--smooth N`)

Raw per-frame angles will jitter. When `N > 0`, apply a centred moving average of window `N` to the **displayed** angle only (the JSON/CSV keep raw values and add an `angle_smoothed_deg` column). Because multiple legs per frame have no identity, match legs between frames by nearest box centre (max distance about 15% of frame width); if no match, start a new history. Keep this simple. A basic nearest-centre matcher is enough. Ultralytics' `model.track(persist=True)` is an acceptable alternative if you prefer; document whichever you choose.

---

## 10. Quality requirements

- **Never crash on a frame with no detection.** Write the frame through unchanged.
- **Memory:** constant memory regardless of video length (stream everything).
- **Progress:** show a `tqdm` bar with frames/sec.
- **Reproducibility:** no randomness.
- **Logging:** print a short summary at the end: frames processed, frames with ≥1 detection, mean angle, output paths, total runtime.
- **Type hints and docstrings** on all public functions. Keep functions small and single-purpose.
- Code must run on CPU-only machines (slowly is fine).

---

## 11. Tests

1. `test_geometry.py`: the reference table in §3 plus the edge cases.
2. `test_detect_schema.py`: load `keypoints.json` fixtures (from the user) and verify the parsing logic handles the schema (3 keypoints per leg, angle computed from kp order lower -> middle -> upper).
3. `test_pipeline_smoke.py`: generate a tiny synthetic video (e.g. 10 frames of 320×240 noise) with `cv2.VideoWriter`, run the pipeline with a **mocked** model that returns one fixed leg detection per frame, and assert that: the output file exists, has the same frame count and resolution, and the JSON/CSV have 10 frames' worth of rows.

---

## 12. Acceptance criteria

The build is done when:

- [ ] `python main.py sample.mp4 --weights best.zip` produces `sample_annotated.mp4`, `sample_annotated_keypoints.json`, and `sample_annotated_angles.csv`.
- [ ] The output video has the same resolution, frame count, and (approximately) duration as the input.
- [ ] Every detected leg shows: blue box with `leg <conf>`, red/yellow/white keypoints, connecting lines, an angle arc, and the angle in degrees.
- [ ] The three reference angles in §3 are reproduced within ±0.2°.
- [ ] Frames with no detection pass through without errors.
- [ ] `--save-frames` writes the raw sliced frames to disk.
- [ ] `pytest` passes.
- [ ] `README.md` explains install, usage, flags, output files, and the keypoint-order/angle convention.

---

## 13. Assumptions to confirm (state any deviations in the README)

1. Keypoint order is **lower -> middle -> upper** (red, yellow, white), as seen in the sample images.
2. The angle of interest is the interior angle at the **middle** keypoint.
3. The model is an Ultralytics YOLO pose model. If `ultralytics` fails to load `best.pt`, print the exact error and the installed `ultralytics` version so the user can match versions.
4. The output is one annotated video, not a side-by-side comparison.
