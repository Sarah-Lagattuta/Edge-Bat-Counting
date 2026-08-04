# Sage Bat Counter

Real-time thermal bat detection and population monitoring on GPU-enabled edge computing devices using **YOLOv11 + SORT tracking**.

Sage Bat Counter adapts the original thermal-video bat counting pipeline developed in:

https://github.com/Sarah-Lagattuta/Bat-Counting-YOLOv11-SORT

and extends it for deployment on **Sage/Waggle edge computing infrastructure** with NVIDIA Thor GPU acceleration.

The system performs real-time thermal bat detection using **YOLOv11**, tracks individual bats across consecutive frames using **SORT (Simple Online and Realtime Tracking)**, and generates automated bat population measurements directly on edge hardware.

By moving inference and tracking from a remote workstation to the deployment device, Sage Bat Counter eliminates the need for researchers to manually collect, transfer, and process large thermal video recordings.

---

# Overview

Traditional thermal bat monitoring workflows require:

1. Recording thermal camera footage in the field
2. Retrieving and transferring large video files
3. Running detection and tracking pipelines offline
4. Reviewing generated results manually

This workflow creates challenges for long-term monitoring because thermal recordings can require significant storage, bandwidth, and processing resources.

Sage Bat Counter moves the complete processing pipeline directly onto Sage/Waggle edge nodes.

The deployed edge system performs:

1. Thermal camera acquisition
2. Thermal frame preprocessing
3. Background subtraction
4. YOLOv11 bat detection
5. Region-of-interest (ROI) filtering
6. SORT multi-object tracking
7. Unique bat counting
8. Publishing measurements through the Sage platform

Instead of transferring complete thermal recordings, the edge device processes video locally and transmits only lightweight count measurements.

This enables autonomous, long-term bat monitoring where edge nodes can collect nightly population data with minimal communication overhead.

---

# Pipeline

The Sage Bat Counter edge pipeline continuously processes thermal imagery on a Sage/Waggle node:

```bash
Thermal Camera
|
v
Frame Capture
|
v
Background Subtraction
|
v
YOLOv11 Detection
|
v
ROI Filtering
|
v
SORT Tracking
|
v
Unique Bat Counting
|
v
Sage Data API
|
v
Nightly Population Dataset
```

The edge device performs all detection and tracking locally.

Rather than storing and transmitting full thermal videos, the system publishes lightweight bat count measurements through the Sage measurement pipeline:

```bash
env.count.bat = <number_of_detected_bats>
```

Measurements are sent through PyWaggle to the local Sage message broker running on the edge node. The Sage runtime then forwards these measurements to the Sage cloud platform, where they can be stored, visualized, and analyzed over time.

These measurements can be collected into nightly population datasets for ecological monitoring and long-term analysis.

---

# Project Structure

```bash
mobile-bat-counter/
├── plugin/                         # Sage/Waggle edge deployment plugin
│ ├── app.py                        # Real-time bat counting application
│ ├── Dockerfile                    # GPU-enabled plugin container
│ ├── requirements.txt              # Plugin dependencies
│ ├── sort/
│ │ └── sort.py                     # SORT tracking implementation
│ ├── sort_shim.py                  # Lightweight SORT compatibility layer
│ └── models/
│   └── best.pt                     # YOLOv11 inference weights
│
├── videos/                         # Sample thermal videos for testing
│
├── data/                           # Generated bat count measurements
│ └── nightly_counts.csv            # Local nightly count history
│
├── models/
│ └── PB_noaug/
│   └── weights/
│       └── best.pt                 # Original research model weights
│
├── src/                            # Original offline processing pipeline
│ ├── tracking.py
│ ├── detection.py
│ └── bg_subtract_new.py
│
├── configs/                        # Pipeline configuration files
│ ├── videos.list                   # Video and ROI configuration
│ └── generated/                    # Generated experiment configs
│
├── Dockerfile                      # Development container configuration
├── pixi.toml                       # Offline development environment
├── pixi.lock
├── README.md
├── run_bat_counter.py              # Offline pipeline entry point
└── sage.yaml                       # Sage/Waggle deployment metadata
```

---

# Edge Deployment (Primary Workflow)

The primary deployment workflow runs Sage Bat Counter as a GPU-enabled Sage/Waggle edge plugin.

The plugin is designed to execute continuously on edge hardware, performing thermal video processing, bat detection, tracking, and count publishing without requiring an external processing server.

The validated deployment environment includes:

- **Hardware:** NVIDIA Thor
- **Architecture:** ARM64
- **GPU Acceleration:** NVIDIA CUDA
- **Inference Framework:** PyTorch + YOLOv11
- **Deployment Platform:** Sage/Waggle
- **Container Runtime:** NVIDIA-enabled Podman

The edge node performs all computationally intensive processing locally:

1. Captures thermal imagery
2. Applies preprocessing and background subtraction
3. Runs YOLOv11 inference
4. Filters detections using the configured ROI
5. Tracks bats across frames using SORT
6. Generates unique bat counts
7. Publishes measurements through the Sage platform

Only count measurements are transmitted, significantly reducing bandwidth requirements compared to transferring raw thermal video recordings.

The plugin container was successfully tested using Podman with direct NVIDIA GPU access, validating the complete edge inference pipeline including GPU acceleration, YOLOv11 detection, ROI filtering, SORT tracking, and bat count generation.

For production Sage deployments, the plugin should be launched through `pluginctl run`. This provides the Sage runtime environment required by PyWaggle to communicate with the local Sage message broker and forward measurements to the Sage cloud platform.

The Kubernetes GPU scheduling workflow through `pluginctl` could not be fully validated because the NVIDIA Kubernetes Device Plugin was not active on the test node. Additional details are provided in the Kubernetes GPU Scheduling section.

---

## Prepare the Plugin Directory

The plugin is built using the plugin/ directory as the build context. Before building, copy the required project files into the plugin directory:

```bash
mkdir -p plugin/models
cp models/PB_noaug/weights/best.pt plugin/models/best.pt

mkdir -p plugin/videos
cp -r videos/* plugin/videos/

mkdir -p plugin/sort
cp sort/sort.py plugin/sort/
```

## Build the Plugin

From the repository root, build the Sage/Waggle plugin container:

```bash
sudo pluginctl build plugin/
```

This creates the plugin image:

```bash
10.31.81.1:5000/local/plugin
```

The generated container includes all required components for standalone edge execution:

- CUDA-enabled PyTorch
- YOLOv11 model weights
- SORT tracking implementation
- Background subtraction pipeline
- ROI configuration support
- Sage/Waggle data publishing functionality

The resulting image can be executed directly using NVIDIA GPU access through Podman for testing and validation.

## Test with a Sample Thermal Video

Before connecting a live thermal camera, the complete edge pipeline can be validated using a recorded thermal video.

Run the plugin container with GPU access:

```bash
podman run --rm -it \
  --name bat-counter \
  --device=nvidia.com/gpu=0 \
  10.31.81.1:5000/local/plugin \
  --camera-source videos/P1.1.2_grey.mov \
  --interval 0
```

A successful execution produces output similar to:

```text
Loading YOLO model from /app/models/best.pt onto cuda
torch.cuda.is_available()=True
...
PUBLISHED env.count.bat = X
Video ended (EOF)
Final unique bat count: X
Saved nightly count: X bats -> data/nightly_counts.csv
```

A successful run verifies the complete edge workflow:

- NVIDIA GPU availability
- CUDA-enabled PyTorch execution
- YOLOv11 inference
- Thermal frame preprocessing
- Background subtraction
- Automatic ROI loading
- SORT tracking
- Unique bat counting
- Nightly measurement generation

The generated count is stored inside the running container:

```bash
data/nightly_counts.csv
```

For persistent storage outside the container, mount a host directory:

```bash
-v $(pwd)/data:/app/data
```

Example output:

```bash
timestamp,bat_count
2026-07-25T04:08:20.539841,1
```

This allows edge nodes to maintain historical bat activity records while avoiding the storage requirements of full thermal video archives.

# Camera Sources

Sage Bat Counter supports multiple input sources depending on the deployment scenario.

The plugin can process:

- Sage/Waggle camera streams
- RTSP network camera streams
- Recorded thermal video files

## 1. Sage/Waggle Camera (Production)

Production deployments use Sage/Waggle camera identifiers.

Example:

```bash
--camera-source bottom_camera
```

The plugin automatically resolves the camera stream through the Sage/Waggle camera interface.

This is the recommended configuration for autonomous field deployments where the edge node continuously monitors bat activity.

## 2. RTSP Camera

Thermal cameras that provide an RTSP stream can be used directly.

Example:

```bash
--camera-source rtsp://camera-address/stream
```

Example deployment:

```bash
sudo pluginctl run --name bat-counter <image> -- \
--camera-source rtsp://user:password@camera-ip:554/stream
```

This mode allows the plugin to process network-connected thermal cameras without requiring local video files.

## 3. Local Thermal Video (Testing)

Recorded thermal videos can be used to validate the pipeline before deploying with a live camera.

Example:

```bash
--camera-source videos/P1.1.2_grey.mov
```

For shorter validation runs, processing can be limited:

```bash
--max-frames <number-of-frames>
```

Example:

```bash
--max-frames 200
```

Local video testing uses the same detection, tracking, and counting pipeline as live Sage/Waggle deployments, allowing deployment behavior to be verified before field installation.

# Plugin Configuration

The Sage Bat Counter plugin exposes configuration parameters that control camera input, inference behavior, tracking settings, and data publishing.

| Parameter | Default | Description |
|---|---|---|
| `camera-source` | `bottom_camera` | Input source: Sage/Waggle camera, RTSP stream, or thermal video file |
| `interval` | `1` | Time between frame captures in seconds |
| `frame-skip` | `0` | Number of frames skipped between YOLO inference runs (`0` = every frame, `1` = every other frame) |
| `weight` | `/app/models/best.pt` | Path to YOLOv11 model weights |
| `confidence` | `0.10` | YOLO detection confidence threshold |
| `imgsz` | `1280` | YOLO inference image resolution |
| `roi` | `0.0 0.0 1.0 1.0` | Manual ROI override if no configured ROI exists |
| `background-subtraction` | `true` | Enable thermal background subtraction preprocessing |
| `bg-window` | `30` | Background subtraction history window size |
| `sort-max-age` | `30` | Maximum number of frames SORT tracks remain active without detections |
| `sort-min-hits` | `5` | Minimum detections required before confirming a track |
| `publish-summary-interval` | `30` | Interval for publishing count summaries |
| `max-frames` | `0` | Stop processing after a fixed number of frames (`0` = unlimited) |

---

# Automatic ROI Configuration

The edge plugin automatically loads deployment-specific regions of interest (ROI) from:

```bash
configs/videos.list
```

Each configuration entry follows the format:

```bash
<location>|<video filename>|<normalized ROI>
```

Example:

```bash
PB|P1.1.2_grey.mov|0.0 0.26 0.97 0.69
```

The ROI values represent normalized coordinates:

```bash
x_min y_min x_max y_max
```

When a thermal video source is provided, the plugin automatically identifies the matching configuration entry and applies the correct ROI during inference.

Example:

```bash
podman run --rm -it \
  --device=nvidia.com/gpu=0 \
  10.31.81.1:5000/local/plugin \
  --camera-source videos/P1.1.2_grey.mov \
  --interval 0
```

The user does not need to manually specify:

```bash
--roi "0.0 0.26 0.97 0.69"
```

Instead, the plugin loads the configured ROI automatically:

```text
Loaded ROI from config for P1.1.2_grey.mov: 0.0 0.26 0.97 0.69
Using ROI from config for videos/P1.1.2_grey.mov: 0.0 0.26 0.97 0.69
```

This allows each deployment location to be configured once and reused for automated monitoring.

## Edge Optimization: Frame Skipping and Inference Tracking

Edge hardware has limited computational resources compared to traditional servers. Sage Bat Counter includes frame skipping controls to reduce GPU workload while maintaining tracking performance.

The plugin tracks two separate quantities:

- Total captured frames
- Frames processed by YOLO inference

This provides visibility into how much computation is performed during deployment.

Example:

```bash
--frame-skip 0
```

Processes every captured frame.

```bash
--frame-skip 1
```

Processes every other frame.

```bash
--frame-skip 2
```

Processes every third frame.

Example runtime output:

```text
frame=200 inference_frames=67 detections=0 tracked=0 unique=1 infer=31.8ms
```

Frame skipping allows deployments to balance:

- GPU utilization
- Power consumption
- Processing throughput
- Detection frequency

This provides flexibility for deployments with different hardware constraints or monitoring requirements.

## Background Subtraction Tradeoff

Background subtraction improves bat detection by removing static thermal scene information before YOLO inference.

The current YOLOv11 model was trained using background-subtracted thermal imagery, so background subtraction is enabled by default.

Enable background subtraction:

```bash
--background-subtraction true
```

Advantages:

- Higher detection consistency
- Matches the model training pipeline
- Recommended for production monitoring

Tradeoffs:

- Additional preprocessing overhead
- Slightly increased latency
- Increased computational requirements

For deployments where throughput or power efficiency is prioritized, background subtraction can be disabled:

```bash
--background-subtraction false
```

Advantages:

- Faster frame processing
- Reduced preprocessing cost
- Lower GPU workload

Tradeoffs:

- Input distribution differs from model training data
- Detection accuracy may decrease depending on thermal conditions

The default configuration keeps background subtraction enabled because it provides the closest match to the training environment of the YOLOv11 model.

---

# Sage/Waggle Deployment

Sage Bat Counter is designed to operate continuously as a Sage/Waggle edge plugin.

The intended deployment workflow is:

1. Build the plugin container
2. Deploy the container on a Sage/Waggle node
3. Connect the thermal camera source
4. Run autonomous bat monitoring
5. Publish nightly bat count measurements

The edge node performs all detection and tracking locally, allowing long-term monitoring without transferring large thermal video files.

## Build

Build the plugin image:

```bash
sudo pluginctl build plugin/
```

## Running the GPU Container

The validated deployment workflow uses the NVIDIA container runtime through Podman:

```bash
podman run --rm -it \
  --name bat-counter \
  --device=nvidia.com/gpu=0 \
  10.31.81.1:5000/local/plugin \
  --camera-source bottom_camera \
  --interval 0
```

The plugin performs:

- Thermal image acquisition
- YOLOv11 inference
- Bat tracking
- Unique count generation
- Sage measurement publishing

Counts are published through PyWaggle using:

```bash
env.count.bat
```

Example measurement:

```bash
env.count.bat = 12
```

The Sage measurement path is:

```text
Sage Bat Counter Plugin
|
v
PyWaggle
|
v
Local Sage Message Broker
|
v
Sage Cloud Platform
|
v
Dashboard Visualization
```

PyWaggle does not directly send data to the internet. Instead, it communicates with the local Sage broker running on the node. When deployed through `pluginctl run`, the Sage runtime automatically provides the required broker configuration and authentication.

The edge device transmits only count information rather than full thermal recordings, reducing bandwidth and storage requirements.

Example long-term monitoring workflow:

```bash
Night 1  -> env.count.bat = 12
Night 2  -> env.count.bat = 18
Night 3  -> env.count.bat = 9
          ...
              |
              v
      Nightly Bat Population Trends
```

## Sage Runtime and Data Publishing

Sage Bat Counter relies on the Sage runtime environment to publish measurements to the Sage cloud platform.

The complete measurement flow is:

```text
Plugin
|
v
PyWaggle Measurement
|
v
Local Sage Broker
|
v
Sage Cloud
|
v
Dashboard
```

PyWaggle packages each measurement with metadata including:

- Measurement key (`env.count.bat`)
- Timestamp
- Measurement value
- Node identity information

The Sage infrastructure handles forwarding, authentication, and associating measurements with the correct edge node.

For this reason, production deployments should use `pluginctl run`, which automatically provides the environment variables required for PyWaggle communication with the local broker.

Standalone Podman execution is useful for validating GPU inference and the bat counting pipeline, but it does not automatically provide the Sage runtime communication layer required for dashboard publishing.

## Kubernetes GPU Scheduling Note

The intended Sage/Waggle deployment method uses `pluginctl` for scheduling workloads onto edge nodes.

The expected deployment workflow is:

```bash
sudo pluginctl run \
  --name sage-bat-counter \
  --selector kubernetes.io/hostname=<sage-node-hostname> \
  --resource nvidia.com/gpu=1 \
  <registry>/<image-name> \
  -- \
  --camera-source bottom_camera
```

However, during testing, the node did not advertise GPU resources to Kubernetes.

The expected GPU resource:

```text
nvidia.com/gpu: 1
```

was unavailable because the NVIDIA Kubernetes Device Plugin was not active on the node.

As a result, Kubernetes could not schedule workloads requesting:

```bash
--resource nvidia.com/gpu=1
```

The plugin container itself was successfully validated using Podman with direct NVIDIA GPU access.

Podman validation confirms that the edge AI pipeline executes correctly. Full Sage cloud publishing requires deployment through the Sage runtime environment so that PyWaggle can communicate with the node's local broker.

---

# Nightly Data Collection

Sage Bat Counter is designed for autonomous long-term bat monitoring.

During execution, the plugin records bat counts locally:

```bash
data/nightly_counts.csv
```

For deployments requiring persistent storage across container restarts, mount a host directory or Sage storage volume:

```bash
-v $(pwd)/data:/app/data
```

This allows each edge node to maintain a lightweight historical record of bat activity without storing large thermal video files.

Example:

```text
timestamp,bat_count
2026-07-25T05:49:02.532,1
```

The edge device continuously collects population measurements and publishes:

```bash
env.count.bat
```

through the Sage platform while maintaining a local backup of collected measurements.

This enables automated nightly monitoring without requiring researchers to manually download and process recordings.

---

# Original Offline Pipeline

Sage Bat Counter preserves the original offline processing workflow from the research pipeline.

The offline workflow remains useful for:

- Reproducing previous experiments
- Evaluating model performance
- Processing historical thermal recordings
- Comparing edge and offline results

The original offline pipeline:

```bash
Thermal Video Recording
|
v
YAML Configuration
|
v
Background Subtraction
|
v
YOLOv11 Detection
|
v
SORT Tracking
|
v
Annotated Video + CSV Results
```

Run the offline pipeline with:

```bash
pixi run python run_bat_counter.py \
--config configs/generated/PB_noaug_PB_P1.2.2_grey.mov_BGon_ROIon.yaml
```

The offline pipeline provides a research and validation workflow, while the Sage/Waggle plugin serves as the primary deployment pathway.

# Development Environment

Docker and Pixi are provided for development, testing, and reproducing the original research environment.

The production Sage/Waggle deployment uses the GPU-enabled plugin container:

```bash
sudo pluginctl build plugin/
```

The production container includes NVIDIA CUDA support through the container runtime.

## Sage Camera Configuration

For deployment, the default camera source should use the Sage/Waggle camera interface.

### `plugin/app.py`

Change:

```python
default=os.environ.get("CAMERA_SOURCE", "videos/P1.1.2_grey.mov")
```

to

```python
default=os.environ.get("CAMERA_SOURCE", "bottom_camera")
```

### sage.yaml

Change:

```yaml
default: "videos/P1.1.2_grey.mov"
```

to

```yaml
default: "bottom_camera"
```

---

## Development Container

Docker can be used to create a reproducible development environment containing:

- CUDA
- PyTorch
- YOLOv11
- OpenCV
- Computer vision dependencies

Build the development image:

```bash
docker build -t bat-count-edge .
```

Start the container:

```bash
docker run -it \
  --name bat-count-edge \
  --device=nvidia.com/gpu=0 \
  -v $(pwd):/workspace/mobile-bat-counter \
  bat-count-edge
```

Verify the environment:

```bash
pixi run python -c \
"import torch, ultralytics, cv2; \
print(torch.__version__); \
print(torch.cuda.is_available()); \
print(ultralytics.__version__); \
print(cv2.__version__)"
```

---

## Running Without Docker

Pixi can install and manage the offline research environment directly.

Install dependencies:

```bash
pixi install
```

Run the offline pipeline:

```bash
pixi run python run_bat_counter.py \
--config configs/generated/PB_noaug_PB_P1.2.2_grey.mov_BGon_ROIon.yaml
```

This environment is maintained for offline experimentation and comparison against edge deployment results.

---

# Performance

The Sage Bat Counter edge pipeline was tested on NVIDIA Thor hardware using GPU-accelerated inference.

Measured performance:

| Metric | Result |
|---|---|
| Device | NVIDIA Thor |
| Architecture | ARM64 |
| Model | YOLOv11n |
| Framework | PyTorch + CUDA |
| GPU acceleration | Enabled |
| Input | Thermal video |
| YOLO inference time | ~30 ms/frame |
| Frame skipping | Configurable |
| Processing mode | Real-time capable |

Example runtime output:

```text
Model loaded. torch.cuda.is_available()=True

frame=100 inference_frames=100 detections=0 tracked=0 unique=0 infer=29.0ms
frame=200 inference_frames=200 detections=0 tracked=0 unique=1 infer=28.8ms

Final unique bat count: 1
Saved nightly count: 1 bats -> data/nightly_counts.csv
```

Example edge validation run:

```text
Camera source: videos/P1.1.2_grey.mov
Frames captured: 529
Inference frames: 529
GPU: NVIDIA Thor CUDA acceleration
Inference time: ~30 ms/frame
Processing speed: ~10 FPS
Final unique bat count: 6
Saved nightly count: 6 bats -> data/nightly_counts.csv
```

The validated edge deployment pipeline successfully performs:

- GPU-accelerated YOLOv11 inference
- Thermal video processing
- Background subtraction
- ROI filtering
- SORT tracking
- Configurable inference frame skipping
- Automated nightly count collection
- Sage data publishing

The complete edge pipeline has been validated on NVIDIA Thor hardware using the NVIDIA container runtime.

Kubernetes-based GPU deployment requires NVIDIA Device Plugin support on the target node.

# Model Information

The YOLO model used by Sage Bat Counter:

```bash
plugin/models/best.pt
```

is based on the original PB_noaug model from:

https://github.com/Sarah-Lagattuta/Bat-Counting-YOLOv11-SORT

The model was trained for thermal bat detection.

---

# Credits

Original pipeline:

**Sarah-Lagattuta/Bat-Counting-YOLOv11-SORT**

https://github.com/Sarah-Lagattuta/Bat-Counting-YOLOv11-SORT

Adapted for edge deployment as part of the NSF Center for Pandemic Insights project.

---

# Notes

- Large thermal video files are excluded from version control.
- The plugin is optimized for GPU-enabled Sage/Waggle edge deployment.
- The original offline pipeline remains available for research validation and comparison.
- The primary output of the edge plugin is Sage bat count measurements rather than annotated video files.
- The plugin container includes model weights, ROI configuration, and test data required for standalone execution.
