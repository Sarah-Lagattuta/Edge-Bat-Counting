# Edge-Bat-Counting

<img width="288" height="205.8" alt="image" src="https://github.com/user-attachments/assets/fb2843a3-4b14-4789-a6f4-ca8ac8446569" /> <img width="308.7" height="205.8" alt="image" src="https://github.com/user-attachments/assets/ef0f0ea3-fb66-43b3-9b2a-d650d7e0e9fb" />
<img width="197.5" height="205.8" alt="Screenshot 2026-07-01 at 4 38 13 PM" src="https://github.com/user-attachments/assets/ac18267c-f295-4cb1-862d-f1ea89ad29a5" />

This project enables bat detection and counting from thermal video through GPU-enabled SAGE Grande edge computing devices.

The model pipeline uses **YOLOv11** (You Only Look Once), an image detection deep learning model, and **SORT (Simple Online and Realtime Tracking)**, a multi-object tracking algorithm, to recognize and track bat flight trajectories. Each unique trajectory recognized by the model in a given video is counted as one bat. This model pipeline is extended for deployment on **Sage/Waggle edge computing infrastructure** with NVIDIA Thor GPU acceleration.

Bat population counts are used widely for conservation management and wildlife disease research. By moving inference and tracking from a remote workstation to the deployment device, Sage Bat Counter eliminates the need for researchers to manually collect, transfer, and process large thermal video recordings.

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

# Project Structure

**update this

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
│       └── best.pt                 # Original YOLOv11 model weights (legacy)
│
├── src/                            # Original offline processing pipeline (legacy)
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
