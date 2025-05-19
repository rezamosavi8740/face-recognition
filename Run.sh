#!/bin/bash
set -e  # Exit if any command fails

echo "📁 DATA_ORIGINAL: $DATA_ORIGINAL"
echo "📁 DATA_ALIGNED: $DATA_ALIGNED"
echo "📁 DATA_TRAIN_FORMAT: $DATA_TRAIN_FORMAT"

# Check required directories
if [ ! -d "$DATA_ORIGINAL" ]; then
    echo "❌ ERROR: DATA_ORIGINAL directory does not exist: $DATA_ORIGINAL"
    exit 1
fi

mkdir -p "$DATA_ALIGNED"
mkdir -p "$DATA_TRAIN_FORMAT"

if [[ -z "$DO_CONVERT" ]]; then
    read -p "❓ Do you want to prepare training data from original images? (y/n): " DO_CONVERT
fi

if [[ "$DO_CONVERT" == "y" || "$DO_CONVERT" == "Y" ]]; then
    echo "🚀 Aligning original images per identity and image..."
    for identity_dir in "$DATA_ORIGINAL"/*; do
        if [ -d "$identity_dir" ]; then
            identity_name=$(basename "$identity_dir")
            output_identity_dir="$DATA_ALIGNED/$identity_name"
            mkdir -p "$output_identity_dir"
            echo "🔧 Aligning identity: $identity_name"

            for img in "$identity_dir"/*.{jpg,JPG,jpeg,JPEG,png,PNG}; do
                if [ -f "$img" ]; then
                    echo "➡️ Aligning image: $img"
                    python /workspace/face-recognition/aligner/aligner.py "$img" --out_dir "$output_identity_dir" \
                        && echo "✅ Successfully aligned: $img" \
                        || echo "❌ Failed to align: $img"
                fi
            done
        fi
    done

    echo "📸 Checking aligned images..."
    find "$DATA_ALIGNED" -type f

    echo "🚀 Converting aligned data to training format (.rec)..."
    CONVERT_LOG=$(python /workspace/face-recognition/codes/CVLface/cvlface/data_utils/recognition/training_data/bundle_images_into_rec.py \
        --source_dir "$DATA_ALIGNED" \
        --out_dir "$DATA_TRAIN_FORMAT")

    echo "$CONVERT_LOG"

    NUM_CLASSES=$(echo "$CONVERT_LOG" | grep "Num unique labels" | grep -o '[0-9]\+')
    NUM_IMAGES=$(echo "$CONVERT_LOG" | grep "Found" | grep -o '[0-9]\+')
else
    echo "ℹ️ Skipping data conversion."
    if [[ -z "$NUM_CLASSES" ]]; then
        read -p "🧮 Enter number of classes: " NUM_CLASSES
    fi
    if [[ -z "$NUM_IMAGES" ]]; then
        read -p "🧮 Enter number of images: " NUM_IMAGES
    fi
fi

if [[ -z "$BATCH_SIZE" ]]; then
    read -p "🧩 Enter batch size for training: " BATCH_SIZE
fi

echo "🔧 Generating training config YAML..."

FINAL_PATH=/workspace/face-recognition/codes/CVLface/cvlface/research/recognition/code/run_v1/dataset/configs/data_final.yaml

cat > "$FINAL_PATH" <<EOF
data_root: '$DATA_ALIGNED'
rec: '$DATA_TRAIN_FORMAT'
color_space: 'RGB'
num_classes: $NUM_CLASSES
num_image: $NUM_IMAGES
repeated_sampling_cfg: null
semi_sampling_cfg: null
EOF

echo "✅ Config written to: $FINAL_PATH"

echo "🚀 Starting training with batch size $BATCH_SIZE..."
LIGHTING_TESTING=1 CUDA_VISIBLE_DEVICES=0 python /workspace/face-recognition/codes/CVLface/cvlface/research/recognition/code/run_v1/train.py \
  trainers=configs/default.yaml \
  trainers.num_gpu=1 \
  trainers.batch_size=$BATCH_SIZE \
  trainers.gradient_acc=1 \
  trainers.num_workers=16 \
  trainers.precision='32-true' \
  trainers.float32_matmul_precision='high' \
  dataset=configs/data_final.yaml \
  data_augs=configs/basic_v1.yaml \
  models=iresnet/configs/v1_ir101.yaml \
  pipelines=configs/train_model_cls.yaml \
  evaluations=configs/quick.yaml \
  classifiers=configs/partial_fc.yaml \
  optims=configs/cosine.yaml \
  losses=configs/adaface.yaml \
  trainers.skip_final_eval=False

echo "🎉 Training complete."
exec bash
