#!/bin/bash

if (( $# != 2 )); then
    >&2 echo "Usage: bash voc_sampler.sh <exp_name> <voc_class_split=1, 2 or 3>"
    exit
fi

EXP_NAME=$1
VOC_CLASS_SPLIT=$2

SAMPLE_POOL_SIZE=100
SAMPLE_OUT_SIZE="1 5 10"
ABLATION=0
RNGSEED=0

for ((i=0; i<10; i++)); do
  python sampler.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${VOC_CLASS_SPLIT}.yaml \
  --sample_pool_size=$SAMPLE_POOL_SIZE --sample_out_size="$SAMPLE_OUT_SIZE" --ablation=$ABLATION \
  --prev_dataseed $i --new_dataseed  $((30+i)) \
  --opts OUTPUT_DIR sampler_logs/voc/${EXP_NAME}/seed$((30+i)) TEST.PCB_MODELPATH  "./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth" \
   SEED $RNGSEED
done


ZIP_NAME="voc${VOC_CLASS_SPLIT}_${EXP_NAME}_pool${SAMPLE_POOL_SIZE}"
if [[ $ABLATION == 1 ]]; then
  ZIP_NAME+="_ablation"
fi
ZIP_NAME+="_rng${RNGSEED}"

zip -r "$ZIP_NAME.zip" datasets/vocsplit/seed3?
mkdir -p voc_splits
mv $ZIP_NAME voc_splits
