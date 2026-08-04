FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1 ffmpeg libxcb1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)") && \
    TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)") && \
    NP_VER=$(python3 -c "import numpy; print(numpy.__version__)") && \
    printf "torch==${TORCH_VER}\ntorchvision==${TV_VER}\nnumpy==${NP_VER}\n" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2*; \
    pip install --no-cache-dir -c /tmp/constraints.txt opencv-python==4.10.0.84

# Model weights
COPY models/best.pt /app/models/best.pt
ENV BAT_MODEL=/app/models/best.pt

# Test videos and ROI configuration
COPY videos/ /app/videos/
COPY configs/ /app/configs/

# Output directory for counts/logs
RUN mkdir -p /app/data

# Application code
COPY app.py /app/app.py
COPY sort_shim.py /app/sort_shim.py
COPY sort/sort.py /app/sort/sort.py

# Offline model behavior
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python3", "/app/app.py"]