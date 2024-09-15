#!/bin/bash

# SEED is the random generator for detectron2.
# PREV_DATASEED is the split number the novel classes files will be copied from, for a fair comparison
# NEW_DATASEED is the split number where the base classes will be stored.

for ((i=0; i<10; i++)); do
    python BaseProtoSampler.py --num-gpus 1 --config-file configs/coco/defrcn_det_r101_base.yaml \
    --opts OUTPUT_DIR sampler_logs/coco/seed$((10+i)) TEST.PCB_MODELPATH  "./ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth" \
     SEED 0 PREV_DATASEED $i NEW_DATASEED $((10+i)) SAMPLE_SIZE 2 SAMPLE_POOL_SIZE 100
done
