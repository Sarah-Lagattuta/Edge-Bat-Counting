# Edge-Bat-Counting

<img width="288" height="205.8" alt="image" src="https://github.com/user-attachments/assets/fb2843a3-4b14-4789-a6f4-ca8ac8446569" /> <img width="308.7" height="205.8" alt="image" src="https://github.com/user-attachments/assets/ef0f0ea3-fb66-43b3-9b2a-d650d7e0e9fb" />
<img width="197.5" height="205.8" alt="Screenshot 2026-07-01 at 4 38 13 PM" src="https://github.com/user-attachments/assets/ac18267c-f295-4cb1-862d-f1ea89ad29a5" />

This project enables bat detection and counting from thermal video through GPU-enabled SAGE Grande edge computing devices.

The model pipeline uses **YOLOv11** (You Only Look Once), an image detection deep learning model, and **SORT (Simple Online and Realtime Tracking)**, a multi-object tracking algorithm, to recognize and track bat flight trajectories. Each unique trajectory recognized by the model in a given video is counted as one bat. This model pipeline is extended for deployment on **Sage/Waggle edge computing infrastructure** with NVIDIA Thor GPU acceleration.

Bat population counts are used widely for conservation management and wildlife disease research. By moving inference and tracking from a remote workstation to the deployment device, Sage Bat Counter eliminates the need for researchers to manually collect, transfer, and process large thermal video recordings.

---

# Project Structure

This project is organized into two parent directories. Each of these contains its own README.md file.

`Original_Pipeline/` contains the base data and information for running the bat counting YOLOv11/SORT model on a local computer for pre-recorded thermal video.

`SAGE_Bat_Counter/` contains all code and data required for running the bat counting model using SAGE edge computing. This directory also contains a copy of the training data and all necessary components from `Original_Pipeline/`.
