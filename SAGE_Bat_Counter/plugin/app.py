"""
Sage/Waggle edge plugin: thermal-video bat counting with YOLO11 + SORT.

Captures frames from a camera (WES-named, RTSP URL, or local file for testing),
runs YOLO11 detection on the GPU for "hot bat" class, tracks with SORT, and
publishes the unique bat count via pywaggle.

Flow:
  camera → frame capture (interval) → YOLO detection (GPU) → SORT tracking → count → pywaggle publish

Reuses the detection+tracking logic from src/tracking.py but adapted for
live streaming + edge deployment.

Usage:
  python app.py --side stream --camera-source bottom_camera --interval 1
  python app.py --side stream --camera-source "rtsp://user:pass@ip/stream" --interval 2
  python app.py --side video --camera-source videos/P1.1.2_grey.mov --interval 0.1   # local test on a file
"""
import argparse
import logging
import os
import sys
import time
import numpy as np

# make the plugin dir importable when running from anywhere
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PLUGIN_DIR)

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bat-counter")

# ─── nightly data logging ────────────────────────────────────────────────

def save_nightly_count(count, output_file="data/nightly_counts.csv"):
    """
    Save bat count measurements locally for long-term monitoring.
    """
    import csv
    from datetime import datetime

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    file_exists = os.path.exists(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "bat_count"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            count
        ])

    logger.info(
        "Saved nightly count: %d bats -> %s",
        count,
        output_file
    )

def load_roi_from_config(source, config_file="configs/videos.list"):
    """
    Load ROI automatically from videos.list.

    Format:
        location|video_source|x1 y1 x2 y2
    """
    if not os.path.exists(config_file):
        logger.info("ROI config not found: %s", config_file)
        return None

    source_name = os.path.basename(source)

    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split("|")

            if len(parts) != 3:
                continue

            _, video, roi = parts

            if video == source_name:
                logger.info(
                    "Loaded ROI from config for %s: %s",
                    source_name,
                    roi
                )
                return roi

    logger.info("No ROI found for %s", source_name)
    return None

# ─── camera helpers ────────────────────────────────────────────────────────

def open_camera(camera_source: str, test_video: bool = False):
    """
    Open a video capture source.

    Sources (in priority order):
      1. WES-named camera (e.g. "bottom_camera") — uses pywaggle Camera on a real node
      2. RTSP URL (e.g. "rtsp://...") — passes directly to cv2.VideoCapture
      3. Local file path (testing) — cv2.VideoCapture on the file

    Returns a cv2.VideoCapture *or* a pywaggle Camera wrapper.
    """
    is_url = "://" in camera_source
    is_file = os.path.exists(camera_source)

    logger.info("Camera source check: %s exists=%s cwd=%s", camera_source, is_file, os.getcwd())

    if is_url and camera_source.startswith("rtsp://"):
        logger.info("Opening RTSP stream: %s", camera_source.split("@")[-1])
        # Force TCP transport — UDP drops packets on high-bitrate 4K HEVC streams,
        # causing cap.read() to return False intermittently.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Failed to open RTSP stream, retrying without TCP override...")
            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
        return cap, "rtsp"

    if is_file or test_video:
        logger.info("Opening video file: %s", camera_source)
        cap = cv2.VideoCapture(camera_source)
        return cap, "file"

    # treat as a WES-named camera → pywaggle Camera
    logger.info("Opening WES-named camera: %s", camera_source)
    try:
        from waggle.data.vision import Camera
        return Camera(camera_source), "wes"
    except Exception as e:
        logger.error(
            "Could not open WES camera '%s' (pywaggle not available or not on a node): %s",
            camera_source, e,
        )
        raise


def grab_frame(cap, source_type: str):
    """Read a single frame from the capture source. Returns (frame BGR ndarray, timestamp_ns)."""
    if source_type == "wes":
        sample = cap.snapshot()
        ts = sample.timestamp
        return sample.data, ts
    # rtsp or file
    ret, frame = cap.read()
    if not ret:
        if source_type == "rtsp":
            logger.warning(
                "RTSP cap.read() returned False (ret=%s, frame=%s) — stream may be "
                "dropping packets, redirecting, or unreachable from this pod",
                ret, None if frame is None else frame.shape,
            )
        return None, None
    ts = time.time_ns()
    return frame, ts


# ─── detection helpers (adapted from src/tracking.py) ─────────────────────

def centered_in_region(box, region, shape):
    """Check if a bounding box center falls within the normalized ROI."""
    box_center = [
        (box[0] + box[2]) / shape[1] / 2.0,
        (box[1] + box[3]) / shape[0] / 2.0,
    ]
    return region[0] < box_center[0] < region[2] and region[1] < box_center[1] < region[3]


def parse_roi(roi_str: str):
    """Parse an ROI string 'x0 y0 x1 y1' (normalized 0-1) into a list of 4 floats."""
    parts = [float(x) for x in roi_str.replace(",", " ").split()]
    if len(parts) != 4 or not all(0.0 <= x <= 1.0 for x in parts):
        raise ValueError(f"ROI must be 4 numbers in [0,1], got: {roi_str!r}")
    return parts


def run_inference(model, frame, amplification, confidence, imgsz):
    """Run YOLO inference, return Nx5 array [x1,y1,x2,y2,conf] for hot-bat detections."""
    results = model(
        frame * amplification,
        verbose=False,
        conf=confidence,
        imgsz=imgsz,
    )[0].cpu()

    detections = results.boxes.xyxy.numpy()
    confidences = results.boxes.conf.numpy()
    class_ids = results.boxes.cls.numpy()

    # only keep hot-bat class (cls == 1)
    hot_mask = class_ids == 1
    detections = detections[hot_mask]
    confidences = confidences[hot_mask]

    return np.column_stack([detections, confidences]) if len(detections) else np.empty((0, 5))


def run_inference_with_roi(model, frame, amplification, confidence, imgsz, roi, region_px):
    """Run YOLO inference + ROI filtering. Returns Nx5 [x1,y1,x2,y2,conf] for SORT."""
    det_conf = run_inference(model, frame, amplification, confidence, imgsz)
    if len(det_conf) == 0:
        return det_conf

    det_xxyy = det_conf[:, :4]
    keep = np.array([
        centered_in_region(d, roi, frame.shape) for d in det_xxyy
    ])
    det_conf = det_conf[keep]
    return det_conf


# ─── optional lightweight background subtraction ──────────────────────────

class RunningBackgroundSubtractor:
    """
    Inline per-frame background subtraction, no full-video transcode.
    Uses a sliding window median like bg_subtract_new.py, but operates
    on recent frames in memory and returns a debackgrounded grayscale frame.

    Disabled by default; enable with --background-subtraction.
    """
    def __init__(self, window_N=30, gamma=0.5):
        from collections import deque
        self.window_N = window_N
        self.buffer = deque(maxlen=window_N)

    def process(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.buffer.append(gray)

        if len(self.buffer) < self.window_N:
            # not enough frames yet — return the raw grayscale expanded to 3ch
            return gray  # caller handles 1ch→3ch if needed

        background = np.median(np.stack(self.buffer), axis=0).astype(np.uint8)
        fg_mask = cv2.absdiff(gray, background)

        # gamma adjustment (from bg_subtract_new.py)
        gamma = 0.5
        table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
        fg_mask = cv2.LUT(fg_mask, table)

        return fg_mask

    def apply_to_frame(self, frame_bgr):
        """Return a debackgrounded 3-channel frame (BGR) suitable for YOLO."""
        fg = self.process(frame_bgr)
        # convert the single-channel fg mask back to 3-channel for YOLO
        return cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)


# ─── main loop ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Bat counting edge plugin (YOLO11 + SORT)", formatter_class=argparse.RawDescriptionHelpFormatter, )
    parser.add_argument("--camera-source", default=os.environ.get("CAMERA_SOURCE", "bottom_camera"), help="WES camera name, RTSP URL, or local file path")
    parser.add_argument("--interval", type=float, default=float(os.environ.get("INTERVAL", "1.0")), help="Seconds between frame captures (0 = read as fast as possible)")
    parser.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame (0 = every frame)")
    parser.add_argument("--weight", default="/app/models/best.pt", help="Path to YOLO model weights")
    parser.add_argument("--confidence", type=float, default=0.10, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--roi", default="0.0 0.0 1.0 1.0", help="ROI as 'x0 y0 x1 y1' normalized 0-1 (tracking region)")
    parser.add_argument("--amplification", type=float, default=1.0, help="Pixel amplification factor before inference")
    parser.add_argument("--background-subtraction", default="true", help="Enable inline background subtraction (true/false). Default true — " "the baked-in PB_noaug weights need bg-subtracted frames to detect bats " "in low-contrast thermal video.")
    parser.add_argument("--bg-window", type=int, default=30, help="Background subtraction sliding window size (frames)")
    parser.add_argument("--sort-max-age", type=int, default=30, help="SORT max_age parameter")
    parser.add_argument("--sort-min-hits", type=int, default=5, help="SORT min_hits parameter")
    parser.add_argument("--sort-iou-threshold", type=float, default=0.10, help="SORT IoU threshold")
    parser.add_argument("--publish-summary-interval", type=int, default=30, help="Publish a summary count every N frames (0 = every frame)")
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("MAX_FRAMES", "0")), help="Stop after N frames (0 = run forever)")
    args = parser.parse_args()

    # --- decide publish mode: pywaggle if available, else log-only for local testing ---
    publish_available = False
    try:
        from waggle.plugin import Plugin
        publish_available = True
    except Exception:
        logger.warning("pywaggle not available — running in local-test mode (publish to stdout)")

    # --- parse ROI ---
    config_roi = load_roi_from_config(args.camera_source)

    if config_roi:
        roi = parse_roi(config_roi)
        logger.info(
            "Using ROI from config for %s: %s",
            args.camera_source,
            config_roi
        )
    else:
        roi = parse_roi(args.roi)
        logger.info(
            "Using ROI from command line: %s",
            args.roi
        )

    logger.info("Using ROI: %s", roi)

    # --- import sort (stubbed skimage/matplotlib via sort_shim) ---
    from sort_shim import Sort
    tracker = Sort(
        max_age=args.sort_max_age,
        min_hits=args.sort_min_hits,
        iou_threshold=args.sort_iou_threshold,
    )

    # --- load model onto GPU if available ---
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading YOLO model from %s onto %s", args.weight, device)
    model = YOLO(args.weight).to(device)
    if hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception:
            pass
    logger.info("Model loaded. torch.cuda.is_available()=%s", torch.cuda.is_available())

    # --- optional background subtractor ---
    bg_sub = None
    if args.background_subtraction.lower() in ("true", "1", "yes", "y"):
        bg_sub = RunningBackgroundSubtractor(window_N=args.bg_window)
        logger.info("Background subtraction ENABLED (window=%d)", args.bg_window)

    # --- open camera ---
    cap, source_type = open_camera(args.camera_source, test_video=args.max_frames > 0)

    # --- metadata for publishing (all string-typed for pywaggle) ---
    safe_source = args.camera_source.replace(" ", "_").replace("/", "_")
    model_name = os.path.basename(args.weight)
    roi_str = args.roi.replace(" ", ",")

    unique_ids = set()
    frame_count = 0
    last_publish_frame = -1
    fps_start = time.time()
    fps_frame_count = 0
    inference_frame_count = 0

    plugin_ctx = None
    plugin = None
    if publish_available:
        plugin_ctx = Plugin()
        plugin = plugin_ctx.__enter__()

    try:
        logger.info(
            "Starting capture loop: source=%s type=%s interval=%.2fs", safe_source, source_type, args.interval
        )

        while True:
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("Reached max_frames=%d, exiting loop", args.max_frames)
                break

            frame, ts = grab_frame(cap, source_type)
            if frame is None:
                if source_type == "file":
                    logger.info("Video ended (EOF)")
                    break
                logger.warning("Frame grab returned None; sleeping and retrying")
                time.sleep(1.0)
                continue

            frame_count += 1
            fps_frame_count += 1

            # skip frames before expensive processing
            if args.frame_skip > 0:
                if frame_count % (args.frame_skip + 1) != 0:
                    continue

            # optional background subtraction
            if bg_sub is not None:
                frame = bg_sub.apply_to_frame(frame)

            # YOLO detection + ROI filter
            infer_start = time.time()
            inference_frame_count += 1

            detections = run_inference_with_roi(
                model, frame, args.amplification, args.confidence,
                args.imgsz, roi, None,
            )
            infer_ms = (time.time() - infer_start) * 1000

            # SORT tracking
            tracked = tracker.update(detections)
            for obj in tracked:
                track_id = int(obj[4])
                unique_ids.add(track_id)

            # periodic logging
            if frame_count <= 5 or frame_count % 100 == 0:
                logger.info(
                    "frame=%d inference_frames=%d detections=%d tracked=%d unique=%d infer=%.1fms",
                    frame_count,
                    inference_frame_count,
                    len(detections),
                    len(tracked),
                    len(unique_ids),
                    infer_ms,
                )

            # publish
            should_publish = (
                args.publish_summary_interval == 0
                or frame_count - last_publish_frame >= args.publish_summary_interval
            )
            if should_publish:
                if publish_available:
                    # all meta values must be strings (pywaggle contract)
                    plugin.publish(
                        "env.count.bat",
                        len(unique_ids),
                        timestamp=ts,
                        meta={
                            "camera": str(safe_source),
                            "model": str(model_name),
                            "roi": str(roi_str),
                            "confidence": str(args.confidence),
                            "imgsz": str(args.imgsz),
                            "amplification": str(args.amplification),
                            "background_subtraction": str(args.background_subtraction),
                            "frame_count": str(frame_count),
                        },
                    )
                    logger.info("PUBLISHED env.count.bat = %d (frame %d)", len(unique_ids), frame_count)
                else:
                    logger.info("PUBLISHED (stdout) env.count.bat = %d (frame %d)", len(unique_ids), frame_count)
                
                last_publish_frame = frame_count

            # FPS reporting every 500 frames
            if fps_frame_count >= 500:
                elapsed = time.time() - fps_start
                logger.info("=== 500 frames in %.1fs (%.1f FPS) ===", elapsed, 500 / elapsed)
                fps_start = time.time()
                fps_frame_count = 0

            # throttle if not a file
            if args.interval > 0 and source_type != "file":
                time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Final unique bat count: %d (after %d frames)", len(unique_ids), frame_count)

        save_nightly_count(len(unique_ids))
        
        if source_type != "wes":
            cap.release()
        if plugin_ctx is not None:
            plugin_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
