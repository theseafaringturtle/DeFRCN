#!/usr/bin/env bash

# Set number of GPUs, training algo etc
source ./deployment_cfg.sh

EXP_NAME=$1
SPLIT_ID=$2

echo "Saving results to "$VOC_BASE_SAVE_DIR

SAVE_DIR=$VOC_BASE_SAVE_DIR/${EXP_NAME}

# ------------------------------- Base Pre-train ---------------------------------- #
if [[ $FINETUNE != true ]] ;
then
  echo "Performing base pretraining"
python3 main.py --num-gpus $NUM_GPUS --config-file configs/voc/defrcn_det_r101_base${SPLIT_ID}.yaml     \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
           OUTPUT_DIR ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
else
  echo "Skipping base pretraining"
  mkdir -p ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
  cp ./model_base_voc/model_final${SPLIT_ID}.pth ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth
fi

# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset voc --method remove                                    \
    --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                      \
    --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_remove.pth

# ------------------------------ Novel Fine-tuning -------------------------------- #
# --> 1. FSRW-like, using seed0 aka default files from TFA. Only one repeat since this is now deterministic across runs.
if [[ $FSRW == true ]]
then
for repeat_id in 0
do
    for shot in $SHOT_LIST
    do
        for seed in 0
        do
            python3 tools/create_config.py --dataset voc --config_root configs/voc \
                --shot ${shot} --seed ${seed} --setting 'fsod' --split ${SPLIT_ID}
            CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}/fsrw-like/${shot}shot_seed${seed}_repeat${repeat_id}
            python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                          \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                   \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} TRAINER $TRAINER
            # Whether you want to keep model checkpoint saved at the end of training
            if [[ $KEEP_OUTPUTS != true ]]; then
              rm ${CONFIG_PATH}
              rm ${OUTPUT_DIR}/model_final.pth
            fi
        done
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}/fsrw-like --shot-list 1 2 3 5 10  # summarize all results
fi

# ----------------------------- Model Preparation --------------------------------- #
if [[ $PROVIDED_RANDINIT == true ]]
then
  # If you want to use same random initialisation for last layers as initial experiment
  if [[ $FINETUNE != true ]]; then echo "FINETUNE is not true, are you sure you want to use the pretrained surgery model?"; fi
  echo "Using provided model_reset_surgery"
  mkdir -p ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
  cp ./model_base_voc/model_reset_surgery${SPLIT_ID}.pth ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth
else
  # Perform it yourself, though seed for randinit will not be the same as initial experiment
  echo "Creating model_reset_surgery.pth"
  python3 tools/model_surgery.py --dataset voc --method randinit                                \
      --src-path ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_final.pth                    \
      --save-dir ${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}
fi
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth


# ------------------------------ Novel Fine-tuning ------------------------------- #
# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
if [[ $GFSOD == true ]]
then
for seed in $SEED_SPLIT_LIST
do
    for shot in $SHOT_LIST
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc               \
            --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} TRAINER $TRAINER
        if [[ $KEEP_OUTPUTS != true ]]; then
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        fi
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10  # summarize all results
fi

# ------------------------------ Novel Fine-tuning ------------------------------- #  not necessary, TFA-like fsod just for the completeness of defrcn results
# --> 3. TFA-like, i.e. run seed0~9 for robust results
if [[ $FSOD_TFA == true ]]
then
BASE_WEIGHT=${SAVE_DIR}/defrcn_det_r101_base${SPLIT_ID}/model_reset_remove.pth
for seed in $SEED_SPLIT_LIST
do
    for shot in $SHOT_LIST
    do
        python3 tools/create_config.py --dataset voc --config_root configs/voc                \
            --shot ${shot} --seed ${seed} --setting 'fsod' --split ${SPLIT_ID}
        CONFIG_PATH=configs/voc/defrcn_fsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}
        python3 main.py --num-gpus $NUM_GPUS --config-file ${CONFIG_PATH}                             \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                      \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} SEED ${seed} TRAINER $TRAINER
        if [[ $KEEP_OUTPUTS != true ]]; then
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        fi
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_fsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10  # summarize all results
fi
echo "End"
