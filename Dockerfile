FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DATA_ALIGNED=/data/output/aligned
ENV DATA_TRAIN_FORMAT=/data/output/train_format

RUN apt-get update && apt-get install -y \
    git curl ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 gettext && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gettext


COPY . /workspace/face-recognition
#COPY facerec_val /workspace/face-recognition/Download/facerec_val
WORKDIR /workspace/face-recognition/codes/CVLface

RUN mkdir -p /root/.pip
COPY pip.conf /root/.pip/pip.conf

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install .
