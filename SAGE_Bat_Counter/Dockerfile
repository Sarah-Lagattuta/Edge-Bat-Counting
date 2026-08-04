FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /workspace/mobile-bat-counter

RUN apt update && apt install -y \
    curl \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

COPY pixi.toml pixi.lock ./

# Install environment during image build
RUN CONDA_OVERRIDE_CUDA=12.9 /root/.pixi/bin/pixi install

COPY . .

ENV CONDA_OVERRIDE_CUDA=12.9

ENTRYPOINT ["/root/.pixi/bin/pixi", "run", "python", "plugin/app.py"]