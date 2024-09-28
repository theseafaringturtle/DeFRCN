#!/bin/bash

if (( $# != 2 )); then
    >&2 echo "Usage: bash voc_sampler.sh <exp_name> <voc_class_split=1, 2 or 3>"
    exit
fi

EXP_NAME=$1
VOC_CLASS_SPLIT=$2

SAMPLE_POOL_SIZE=100
SAMPLE_OUT_SIZE=10
ABLATION=0

for ((i=0; i<10; i++)); do
  python BaseProtoSampler.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${VOC_CLASS_SPLIT}.yaml \
  --sample_pool_size=$SAMPLE_POOL_SIZE --sample_out_size=$SAMPLE_OUT_SIZE --ablation=ABLATION \
  --prev_dataseed $i --new_dataseed  $((30+i)) \
  --opts OUTPUT_DIR sampler_logs/voc/${EXP_NAME}/seed$((30+i)) TEST.PCB_MODELPATH  "./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth" \
   SEED 0
done

if [[ $ABLATION == True ]]; then
  ZIP_NAME="voc_${EXP_NAME}_${SAMPLE_POOL_SIZE}_ablation.zip"
else
  ZIP_NAME="voc_${EXP_NAME}_${SAMPLE_POOL_SIZE}.zip"
fi

zip -r $ZIP_NAME datasets/vocsplit/seed3?
mkdir -p voc_splits
mv $ZIP_NAME voc_splits
