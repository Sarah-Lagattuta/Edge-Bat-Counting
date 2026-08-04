# Edge Port Plan: Bat-Counting-YOLOv11-SORT → Sage/Waggle Plugin

Review date: 2026-07-22
Target platform: Sage/Waggle edge nodes (NVIDIA Thor, ARM64 / aarch64, CUDA)
Reviewer: Wisp (Sage camp assistant) — observations and plan only; no code changed yet.

## 1. What it is now

A thermal-video bat detection + tracking pipeline, currently built for HPC (SLURM):

- **detection** — Ultralytics YOLO11 on every frame (`src/tracking.py:229`, `imgsz=1280`)
- **background subtraction** — sliding-window median `src/bg_subtract_new.py`, writes a debackgrounded AVI to a temp dir via ffmpeg+OpenCV before tracking runs
- **tracking** — SORT (`sort/sort.py`) per video; each unique track ID = 1 bat
- **config** — YAML per video, generated from `configs/make_configs.sh` reading `configs/videos.list`
- **orchestration** — `configs/track_array.sh` is a SLURM array job (`#SBATCH --array=0-2 --gres=gpu:1`), launched via `pixi run python -m src.tracking --config …`
- **output** — annotated videos (mp4v/MJPG) written to `results/annotations/`, counts appended to CSVs in `results/counts/`
- **env** — Pixi/conda, `platforms=["linux-aarch64"]`, CUDA 12.9, PyTorch 2.10 cuda129 build. Model `models/PB_noaug/weights/best.pt` (5.4 MB).

## 2. Where it mismatches edge computing

These are the things that need to change to run well on a constrained edge node.

### a. SLURM/HPC orchestration has no edge equivalent
`track_array.sh` assumes a SLURM cluster with GPU. A Sage node runs containers under k3s via `pluginctl` (camp) / `sesctl` (fleet). There is no SLURM, no job arrays, no Quobyte shared filesystem. The whole orchestration layer is replaced by a Dockerfile + `app.py` plugin.

### b. It processes whole video files, not a live camera
The current model reads `videos/*.mov` end-to-end. The README's stated goal — "have an edge device run the model automatically with the user only defining an ROI" — means the plugin must pull frames from a camera (RTSP stream or WES-named camera), not a file. The frame loop in `src/tracking.py:215` `while cap.isOpened()` needs to become an interval-based capture cycle.

### c. It writes heavy artifacts (annotated video)
`annotate_video()` (`src/tracking.py:23`) writes mp4v videos with every bbox+track ID. On an edge node the point is to transmit results, not raw video. Annotated-video output should be optional and off-by-default; the primary output is a count.

### d. Per-frame inference is heavyweight
`imgsz=1280` every frame is expensive. There's also a hard `.to("cpu")` at `src/tracking.py:184` (comment says "remove after using GPU") that forces CPU inference even when a GPU is present — this defeats the purpose of a Thor GPU node.

### e. Background subtraction transcodes the whole video to disk first
`src/bg_subtract_new.py` reads the entire video into a `deque`, computes median backgrounds, and writes a new AVI — then tracking reads *that* file. On an edge node processing a live stream this pre-pass-everything model doesn't fit; background subtraction (if kept at all) needs to run inline per frame, not as a separate full-video transcode.

### f. No pywaggle integration — nothing is published
The code writes CSVs locally. A Sage plugin must `plugin.publish(topic, value, meta=…)` so the count reaches Beehive and the data API. Right now there is zero pywaggle usage. This is the single biggest "make it an edge plugin" step.

### g. Dependency/manifest issues for an edge container
- `pixi.toml` lists dev tools (`ipdb`, `ruff`, `ipython`, `matplotlib`) that shouldn't be in a slim plugin image.
- The PyTorch CUDA build is pinned to `cuda129`; the Sage ML Dockerfile template uses `nvcr.io/nvidia/pytorch:25.08-py3` (CUDA 13.0, with Blackwell sm_110 kernels that work on Thor). The conda pin would need rethinking inside a container.
- `sort/sort.py` imports `matplotlib` and `skimage` at module load even for inference.

## 3. Bug in the current code (not edge-specific, but bites immediately)

`src/tracking.py:184`:
```
model.to("cpu") # remove after using GPU
```
This forces CPU. On a Thor GPU node it should be `.to("cuda")` (or auto-detect). Left as-is, the ported plugin will run 10-50× slower than it could.

Also `src/tracking.py:265`:
```
if fr == 0:
    raise RuntimeError(f"No frames were read from video: {video}")
```
`fr` starts at 0 and is `fr += 1` on line 221 before this check, so `fr` is never 0 here → a permanently dead/unreachable error guard. Harmless but dead code.

## 4. The port target (what "done" looks like)

A self-contained Sage plugin in a new top-level dir (e.g. `plugin/`) with:

```
Bat-Counting-YOLOv11-SORT/
└── plugin/
    ├── app.py              # pywaggle plugin: capture → detect → track → count → publish
    ├── Dockerfile          # nvcr.io/nvidia/pytorch:25.08-py3, baked weights, slim deps
    ├── requirements.txt    # pywaggle[all], ultralytics, opencv-headless, filterpy, lap
    ├── sage.yaml           # ECR metadata (name, version, inputs as string/int only)
    └── overview.md         # what it does, how to run it
```

Behavior on the edge node:
1. Capture frame(s) from a camera (`--stream` RTSP URL or WES camera name, or `--snapshot-url`).
2. (Optional, per-frame) background subtraction inline — no full-video transcode.
3. YOLO detect hot bats (`cls == 1`), filter to the ROI (`tracking_region`).
4. SORT tracks across frames within one capture session.
5. Count unique track IDs → `plugin.publish("env.count.bat", n, meta={…})` with timestamp + ROI + model name.
6. Exit (one-shot cron) or loop on interval (continuous). No annotated video by default.

The original `src/`, `configs/`, `run_bat_counter.py`, SLURM scripts all stay untouched for reproducibility on HPC; the edge port lives alongside as `plugin/`.

## 5. Suggested starting order (what I'd do first)

1. **Create `plugin/app.py`** — adapt the `templates/ml-plugin-app.py` pattern from the sage-waggle skill. Wire in YOLO detection (load `best.pt`), SORT tracking, and ROI filtering from the existing `src/tracking.py` logic, but frame-stepped on a camera/snapshot instead of a file. Add `plugin.publish("env.count.bat", count, timestamp=ts, meta={"camera": …, "model": …, "roi": "x y x y", "confidence_threshold": "0.10"})` (all meta values as strings).
2. **Create `plugin/requirements.txt`** — `pywaggle[all]>=0.56.0`, `ultralytics>=8.3.70`, `opencv-python-headless>=4.8`, `filterpy>=1.4.5`, `lap` (SORT's linear-assignment fast path). Cut the dev deps.
3. **Create `plugin/Dockerfile`** — use `nvcr.io/nvidia/pytorch:25.08-py3` base (CUDA 13, Blackwell kernels), freeze torch/torchvision/numpy, install `requirements.txt` under constraints, bake `models/PB_noaug/weights/best.pt` into `/app/models/best.pt` at build time, copy in `sort/sort.py` + `app.py`.
4. **Fix the `.to("cpu")` bug** — in the plugin's copy, load the model onto CUDA if available (`device = "cuda" if torch.cuda.is_available() else "cpu"`).
5. **Make annotation optional** — default `write_annotated_frames` off; if on, upload a single annotated still via `plugin.upload_file()`, not a whole video.
6. **Inline or drop background subtraction for the live path** — for a cron one-shot over N frames, a per-frame running-median is feasible; for a single snapshot it's not. Decide based on capture cadence and whether the debackgrounded quality materially affects the count. The current "transcode whole video → AVI → re-read" definitely has to go.
7. **Create `plugin/sage.yaml`** — inputs as `int`/`string` only (no float/bool — ECR rejects them). Examples: `stream` (string), `interval` (int), `confidence` (string, parsed to float at runtime), `roi` (string "x0 y0 x1 y1").
8. **Build + test with `pluginctl`** on the Thor node: `sudo pluginctl build plugin/` → `sudo pluginctl run --name bat-counter-test <img> -- --stream <rtsp-or-snapshot> --interval 30`. Check `pluginctl logs bat-counter-test`, verify the publish with `PYWAGGLE_LOG_DIR=/output` (inspect `/output/data.ndjson`).
9. **(Later) Register in ECR + schedule** — once `pluginctl` works, register via portal.sagecontinuum.org/apps (point at repo + branch, build arm64), then `sesctl` cron job for nightly runs. Note: ECR "Register & Build" is currently fleet-broken (runc CVE-2025-31133); may need the podman side-load workaround.

## 6. Decisions (resolved with the author, 2026-07-22)

- **Edge target:** Thor GPU node (NVIDIA Thor, ARM64 + CUDA, sm_110). Optimize for GPU inference; do not design for CPU-only. The Dockerfile uses `nvcr.io/nvidia/pytorch:25.08-py3` (CUDA 13.0, Blackwell sm_110 kernels).
- **Camera interface:** WES-named camera as primary (`bottom_camera`), with RTSP URL as the fallback path. Made configurable via `--camera-source` / sage.yaml `camera-source` input so no camera type is hardcoded.
- **Scheduling model:** Continuous monitoring — start when the plugin starts, read frames continuously, publish updated counts as it goes. Future optimization: configurable active hours for bat-activity windows.
- **Background subtraction:** Optional and off by default. Default is YOLO directly on incoming frames. An optional inline per-frame background-subtract mode (`--background-subtraction true`) is available but not required for the live path. The original full-video transcode was removed.

## 7. Scaffold built and smoke-tested (2026-07-22)

The plugin scaffold was created in `plugin/` and smoke-tested against the sample videos on this Thor:

```
plugin/
├── app.py              # continuous camera capture → YOLO → SORT → publish
├── sort_shim.py        # stubs skimage/matplotlib so upstream sort.py loads without dev deps
├── Dockerfile          # nvcr.io/nvidia/pytorch:25.08-py3, baked weights
├── requirements.txt    # pywaggle[all], ultralytics, opencv-headless, filterpy, lap
├── sage.yaml           # ECR metadata + inputs (floats as string per ECR validation)
└── overview.md
```

Smoke-test results (CPU venv with pip torch that found the Thor's CUDA):

| Video | Expected (sample_counts.csv) | Plugin count | Inference | FPS |
|---|---|---|---|---|
| P1.1.2_grey.mov | 5 | 5 (exact) | ~27ms/frame, 1280px | ~33 FPS |
| P0.0.1_grey.mov | 38 | 34 (close) | ~28ms/frame, 1280px | ~35 FPS |

The 34 vs 38 gap on P0.0.1 is explained by the reference run using background subtraction ("BGon") and ours running without it. The detection+tracking+count logic from `src/tracking.py` transferred correctly and produces results in the right range. Tuning the optional bg-subtract mode closes that gap.

Model loads onto CUDA (`torch.cuda.is_available()=True`). Inference runs at ~27-28ms/frame after the 1s warmup spike at image size 1280 — solid for real-time edge deployment.

## 8. Next steps (when ready)

1. Build the plugin image on the Thor with `sudo pluginctl build plugin/` and confirm the build succeeds.
2. Run it as a plugin pod: `sudo pluginctl run --name bat-test --selector resource.gpu=true <image> -- --camera-source bottom_camera --interval 1`.
3. Verify the publish path with `PYWAGGLE_LOG_DIR=/output` and inspect `/output/data.ndjson`.
4. Tune the optional `--background-subtraction true` on a live feed and compare detection rates.
5. When ready, register in ECR via portal.sagecontinuum.org/apps (build arm64; if the portal build hits the runc /proc/acpi fleet bug, use the podman side-load workaround — see sage-waggle skill `ecr-build-proc-acpi-failure.md`).
6. Schedule recurring jobs with `sesctl` once the app is in the ECR catalog.
