FROM train_fr:latest

ENV DATA_ORIGINAL=/data/input/original
ENV DATA_ALIGNED=/data/output/aligned
ENV DATA_TRAIN_FORMAT=/data/output/train_format

COPY . /workspace/face-recognition
WORKDIR /workspace/face-recognition/codes/CVLface

RUN chmod +x /workspace/face-recognition/Run.sh

ENTRYPOINT ["/workspace/face-recognition/Run.sh"]
