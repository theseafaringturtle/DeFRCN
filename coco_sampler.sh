#!/bin/bash

if (( $# != 1 )); then
    >&2 echo "Usage: bash coco_sampler.sh <exp_name>"
    exit
fi

EXP_NAME=$1

SAMPLE_POOL_SIZE=100
SAMPLE_OUT_SIZE="1 5 10"
ABLATION=0

for ((i=0; i<10; i++)); do
  python sampler.py --num-gpus 1 --config-file configs/coco/defrcn_det_r101_base.yaml \
  --sample_pool_size=$SAMPLE_POOL_SIZE --sample_out_size="$SAMPLE_OUT_SIZE" --ablation=$ABLATION \
  --prev_dataseed $i --new_dataseed  $((10+i)) \
  --opts OUTPUT_DIR sampler_logs/coco/seed$((10+i)) TEST.PCB_MODELPATH  "./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth" \
   SEED 0
done

if [[ $ABLATION == True ]]; then
  ZIP_NAME="coco_${EXP_NAME}_${SAMPLE_POOL_SIZE}_ablation.zip"
else
  ZIP_NAME="coco_${EXP_NAME}_${SAMPLE_POOL_SIZE}.zip"
fi

zip -r $ZIP_NAME datasets/cocosplit/seed1?
mkdir -p coco_splits
mv $ZIP_NAME coco_splits
