#!/bin/bash
set -e  

echo "📁 DATA_ALIGNED: $DATA_ALIGNED"
echo "📁 DATA_TRAIN_FORMAT: $DATA_TRAIN_FORMAT"

if [ ! -d "$DATA_ALIGNED" ]; then
    echo "❌ ERROR: DATA_ALIGNED directory does not exist: $DATA_ALIGNED"
    exit 1
fi

mkdir -p "$DATA_TRAIN_FORMAT"

echo "🚀 Converting aligned data to training format..."
python codes/CVLface/cvlface/data_utils/recognition/training_data/bundle_images_into_rec.py \
  --data-dir "$DATA_ALIGNED" \
  --output-dir "$DATA_TRAIN_FORMAT" \
  --ext jpg

echo "$CONVERT_LOG"

NUM_CLASSES=$(echo "$CONVERT_LOG" | grep -i "Number of classes" | grep -o '[0-9]\+')
NUM_IMAGES=$(echo "$CONVERT_LOG" | grep -i "Number of images" | grep -o '[0-9]\+')

sed -i "s/^num_classes:.*$/num_classes: $NUM_CLASSES/" "$FINAL_PATH"
sed -i "s/^num_image:.*$/num_image: $NUM_IMAGES/" "$FINAL_PATH"

echo "✅ Set num_classes=$NUM_CLASSES and num_image=$NUM_IMAGES in $FINAL_PATH"

echo "✅ Conversion completed successfully!"

echo "🔧 Generating training config from template..."
TEMPLATE_PATH=codes/CVLface/cvlface/research/recognition/code/run_v1/dataset/configs/data.yaml
FINAL_PATH=codes/CVLface/cvlface/research/recognition/code/run_v1/dataset/configs/data_final.yaml

if [ ! -f "$TEMPLATE_PATH" ]; then
    echo "❌ ERROR: Template YAML file not found: $TEMPLATE_PATH"
    exit 1
fi

envsubst < "$TEMPLATE_PATH" > "$FINAL_PATH"
echo "✅ Config generated at $FINAL_PATH"

# آموزش مدل (اگر بعداً بخوای فعالش کنی)
# echo "🚀 Starting training..."
# python codes/CVLface/cvlface/research/recognition/train.py -c "$FINAL_PATH"

# حذف اختیاری فایل‌های aligned
# echo "🧹 Removing aligned images to save space..."
# rm -rf "$DATA_ALIGNED"

echo "🎉 All done. Training data saved to: $DATA_TRAIN_FORMAT"
