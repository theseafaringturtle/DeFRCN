from __future__ import annotations

import time
import logging
from typing import Dict

import torch
from detectron2.data import MetadataCatalog
from defrcn.dataloader import build_detection_train_loader

from BaseProtoSampler import BaseProtoSampler

logger = logging.getLogger("defrcn").getChild("sampler")


class BaseAblationSampler(BaseProtoSampler):
    """An ablation class that does not create protos and picks samples at random without ranking"""

    def build_prototypes(self, pool_size: int):
        logger.info("Gathering samples...")
        start_time = time.perf_counter()

        memory_cfg = self.cfg.clone()
        memory_cfg.defrost()
        memory_cfg.DATALOADER.SAMPLER_TRAIN = "FiniteTrainingSampler"
        # To obtain 1 image per GPU
        memory_cfg.SOLVER.IMS_PER_BATCH = 1
        memory_loader = build_detection_train_loader(memory_cfg)
        memory_iter = iter(memory_loader)

        for inputs in memory_iter:
            assert len(inputs) == 1

            # We have enough samples to start ranking them, stop going through dataset
            if all([len(v) >= pool_size for k, v in self.class_samples.items()]):
                break
            has_req_classes = []
            file_name = inputs[0]['file_name']
            gt_classes = inputs[0]['instances'].get("gt_classes")

            for c in gt_classes.tolist():
                if len(self.class_samples[int(c)]) < pool_size:
                    has_req_classes.append(True)
                    self.class_samples[int(c)].add(file_name)
                    # Notify when a class's required sample pool has been filled
                    if len(self.class_samples[int(c)]) >= pool_size:
                        logger.info(f"Sample pool for {self.base_class_id_to_name(c)} has been filled")
                    break
            if not any(has_req_classes):
                continue

        logger.info(f"Enough samples ({pool_size}) have been gathered for all classes")
        end_time = time.perf_counter()
        logger.info(f"Sample gathering time: {end_time - start_time} s")

        return {}

    def filter_samples(self, prototypes: Dict, samples_needed: int) -> Dict:
        samples_per_class = {}
        for class_name in self.class_samples.keys():
            samples_per_class[class_name] = list(self.class_samples[class_name])[:samples_needed]
        logger.info(f"Random {samples_needed} samples returned (ablation)")
        return samples_per_class

    def base_class_id_to_name(self, class_id: int):
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if 'voc' in train_set_name:
            base_classes = MetadataCatalog.get(train_set_name).get("base_classes", None)
            return base_classes[class_id]
        elif 'coco' in train_set_name:
            return MetadataCatalog.get(train_set_name).get("base_classes")[class_id]
        else:
            raise Exception(
                "You need to specify a class ID mapping for base classes, or add your dataset to this function")
