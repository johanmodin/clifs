# Start with an nvidia cuda container that supports 3000 series GPUs
FROM nvidia/cuda:11.2.2-base-ubuntu20.04 AS clifs

ARG MODEL=RN50x4

# One of "RN50" < "RN101" < "ViT-B/32" < "RN50x4" (at time of writing)
ENV MODEL=${MODEL}
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
# OpenCV install is here just for dependencies -- we use conda env
RUN apt-get update && apt-get install -y git python3-opencv python3-pip
COPY app/requirements.txt .
RUN pip3 install -r requirements.txt

# Use PyTorch Version 1.8.2
RUN pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Newer version of CLIP adds support for this pytorch version
RUN pip3 install --upgrade git+https://github.com/openai/CLIP.git

# Preload the chosen model -- device doesn't matter here
RUN python3 -c "import clip; clip.load('$MODEL', device='cpu')"
RUN ln -sf /usr/bin/python3 /usr/bin/python
COPY app/ /app
CMD ["python", "clifs.py"]

