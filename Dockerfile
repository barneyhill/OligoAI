# Use RunPod's PyTorch base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Update system packages and install vim
RUN apt-get update && \
    apt-get install -y vim wget curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    flash-attn==2.3.2 \
    pandas \
    scipy \
    lightning \
    wandb \
    ml_collections

# Uninstall torchvision as specified
RUN pip uninstall -y torchvision

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Expose Jupyter port
EXPOSE 8888

# Set default command (RunPod will override this)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
