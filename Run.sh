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

echo "✅ Conversion completed successfully!"


#echo "🧹 Removing aligned images to save space..."
#rm -rf "$DATA_ALIGNED"

echo "🎉 All done. Training data saved to: $DATA_TRAIN_FORMAT"
