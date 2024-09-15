#!/bin/bash

# SEED is the random generator for detectron2.
# PREV_DATASEED is the split number the novel classes files will be copied from, for a fair comparison
# NEW_DATASEED is the split number where the base classes will be stored.

if (( $# != 1 )); then
    >&2 echo "Usage: bash voc_sampler.sh <voc_class_split=1, 2 or 3>"
    exit
fi

VOC_CLASS_SPLIT=$1

for ((i=0; i<10; i++)); do
  python BaseProtoSampler.py --num-gpus 1 --config-file configs/voc/defrcn_det_r101_base${VOC_CLASS_SPLIT}.yaml \
  --opts OUTPUT_DIR sampler_logs/voc/seed$((30+i)) TEST.PCB_MODELPATH  "./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth" \
   SEED 0 PREV_DATASEED $i NEW_DATASEED $((30+i)) SAMPLE_SIZE 2 SAMPLE_POOL_SIZE 100
done
