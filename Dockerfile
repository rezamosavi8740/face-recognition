FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git curl ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/* \

COPY . /workspace/face-recognition
WORKDIR /workspace/face-recognition/codes/CVLface

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install .
