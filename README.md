# Edge-Bat-Counting

<img width="288" height="205.8" alt="image" src="https://github.com/user-attachments/assets/fb2843a3-4b14-4789-a6f4-ca8ac8446569" /> <img width="308.7" height="205.8" alt="image" src="https://github.com/user-attachments/assets/ef0f0ea3-fb66-43b3-9b2a-d650d7e0e9fb" />
<img width="197.5" height="205.8" alt="Screenshot 2026-07-01 at 4 38 13 PM" src="https://github.com/user-attachments/assets/ac18267c-f295-4cb1-862d-f1ea89ad29a5" />

This project enables bat detection and counting from thermal video through GPU-enabled SAGE Grande edge computing devices.

The model pipeline uses **YOLOv11** (You Only Look Once), an image detection deep learning model, and **SORT (Simple Online and Realtime Tracking)**, a multi-object tracking algorithm, to recognize and track bat flight trajectories. Each unique trajectory recognized by the model in a given video is counted as one bat. This model pipeline is extended for deployment on **Sage/Waggle edge computing infrastructure** with NVIDIA Thor GPU acceleration.

Bat population counts are used widely for conservation management and wildlife disease research. By moving inference and tracking from a remote workstation to the deployment device, Sage Bat Counter eliminates the need for researchers to manually collect, transfer, and process large thermal video recordings.

---

# Overall Project Structure

This project is organized into two primary components. Each of these contains its own README.md file.

- **`Original_Pipeline/`** — The original YOLOv11/SORT bat counting pipeline for processing pre-recorded thermal videos on a local workstation. This directory contains the original model, training workflow, and supporting documentation.

- **`SAGE_Bat_Counter/`** — The edge-computing implementation of the pipeline for deployment on SAGE/Waggle hardware with NVIDIA Thor GPU acceleration. This directory contains the deployment code, configuration files, and the resources required to execute the model on edge devices.
