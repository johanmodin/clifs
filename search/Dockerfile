# PyTorch needed for CLIP
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel AS clifs
ARG MODEL=RN50x4

# One of "RN50" < "RN101" < "ViT-B/32" < "RN50x4" (at time of writing)
ENV MODEL=${MODEL}
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
# OpenCV install is here just for dependencies -- we use conda env
RUN apt-get update && apt-get install -y git python3-opencv
COPY app/requirements.txt .
RUN pip install -r requirements.txt

# Preload the chosen model -- device doesn't matter here
RUN python3 -c "import clip; clip.load('$MODEL', device='cpu')"

COPY app/ /app
#CMD ["python", "clifs.py"]
CMD ["/bin/bash"]
