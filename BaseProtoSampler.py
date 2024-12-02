from __future__ import annotations

import glob
import json
import logging
import pickle
import shutil
import time
from contextlib import nullcontext
from typing import Dict, Union, Any, Tuple

import torch
from detectron2.data import MetadataCatalog

import os
import cv2
import torch
import logging
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from defrcn.dataloader import build_detection_test_loader, build_detection_train_loader
from defrcn.evaluation.archs import resnet101

logger = logging.getLogger("defrcn").getChild("sampler")


class BaseProtoSampler:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.PCB_ALPHA

        self.sample_roi_features = {}  # file_name to feature tensor
        self.sample_roi_labels = {}  # file_name to feature labels
        self.class_samples = {cls_id: set() for cls_id in
                              range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}  # class_id to [file_name]

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def build_prototypes(self, pool_size: int):
        logger.info("Gathering samples...")
        start_time = time.perf_counter()
        # Adapted from DeFRCN's PCB code, but using loader for shuffling.
        all_features, all_labels = [], []

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

            file_name = inputs[0]['file_name']
            has_req_classes = []
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

            # Load support images and gt-boxes. Same as PCB.
            img = cv2.imread(file_name)  # BGR
            img_h, img_w = img.shape[0], img.shape[1]
            ratio = img_h / inputs[0]['instances'].image_size[0]
            inputs[0]['instances'].gt_boxes.tensor = inputs[0]['instances'].gt_boxes.tensor * ratio
            boxes = [x["instances"].gt_boxes.clone().to(self.device) for x in inputs]

            # extract roi features
            _features = self.extract_roi_features(img, boxes)
            avg_features, avg_labels = self.average_roi_features(_features, gt_classes)
            self.sample_roi_features[file_name] = avg_features.cpu().clone()
            all_features.append(avg_features.cpu().clone().data)

            self.sample_roi_labels[file_name] = avg_labels.cpu().clone()
            all_labels.append(avg_labels.cpu().clone().data)

        logger.info(f"Enough samples ({pool_size}) have been gathered for all classes")
        end_time = time.perf_counter()
        logger.info(f"Sample gathering time: {end_time - start_time} s")

        # concat
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_features.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            print(f"Creating prototype for class {label} ({self.base_class_id_to_name(label)})")
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)

        return prototypes_dict

    def average_roi_features(self, features, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average features for instances of the same class appearing in the same image.
           This makes images with multiple object instances less over-represented in final prototype"""
        # Get unique labels and their indices
        uniq_labels, indices = torch.unique(labels, return_inverse=True)
        uniq_labels = uniq_labels.to(features.device)
        indices = indices.to(features.device)

        # Use scatter_add to sum features for each unique label
        avg_features = torch.zeros(uniq_labels.size(0), features.size(1), dtype=features.dtype, device=features.device)
        avg_features.scatter_add_(0, indices.unsqueeze(1).expand(-1, features.size(1)), features)

        # Count occurrences of each label
        label_counts = torch.bincount(indices)

        # Divide summed features by label counts to get averages
        avg_features /= label_counts.float().unsqueeze(1)

        return avg_features, uniq_labels

    def filter_samples(self, prototypes: Dict, samples_needed: int) -> Dict:
        samples_per_class = {}
        for class_name in self.class_samples.keys():
            # same_class_dist = []
            # other_class_dist = []
            sim_scores = []
            for file_name in self.class_samples[class_name]:
                sample_features = self.sample_roi_features[file_name]
                # When there are multiple RoI box features in an image, can't average them since they might have different labels
                # Average the ones with the same label instead.
                sample_labels = self.sample_roi_labels[file_name]
                sample_feature_means = {}
                for label in torch.unique(sample_labels):
                    sample_feature_means[label.item()] = sample_features[sample_labels == label].mean(axis=0)
                sample_distances = {}
                for label in sample_feature_means:
                    dist = euclidean_distances(prototypes[label], sample_feature_means[label].unsqueeze(0))
                    sample_distances[label] = dist
                # Create a collated similarity score from different labels
                sim_score = 0.
                for label in sample_distances:
                    if label == class_name:
                        sim_score += sample_distances[label]
                sim_tuple = (file_name, sim_score)
                sim_scores.append(sim_tuple)
            sim_scores.sort(key=lambda tup: tup[1])
            # print(f"Distances: {sim_scores}")
            samples_per_class[class_name] = [file_name for file_name, dist in sim_scores[:samples_needed]]
        logger.info("Samples have been ranked!")
        return samples_per_class

    def extract_roi_features(self, img, boxes):

        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)

        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors.detach()

    def create_dirs(self, prev_seed, new_seed) -> Tuple[str, str]:
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if "voc" in train_set_name:
            base_dir = os.path.join('datasets', "vocsplit")
        elif "coco" in train_set_name:
            base_dir = os.path.join('datasets', "cocosplit")
        else:
            raise Exception("Specify a split directory for your dataset")
        # Get previous directory for novel classes
        prev_seed_dir = os.path.join(base_dir, f"seed{prev_seed}")
        # Create new directory to store new data as a new seed, starting from whichever number was provided in config
        new_seed_dir = os.path.join(base_dir, f"seed{new_seed}")
        os.makedirs(new_seed_dir, exist_ok=True)
        return prev_seed_dir, new_seed_dir

    def save(self, filenames_per_base_class: Dict, samples_needed: int, prev_seed: int, new_seed: int):
        logger.info("Saving split to disk...")
        # Copy novel class files verbatim, since changing those would change the benchmark
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        prev_seed_dir, new_seed_dir = self.create_dirs(prev_seed, new_seed)
        if 'voc' in train_set_name:
            txt_files = os.listdir(prev_seed_dir)
            # Get novel class names
            novel_classes = MetadataCatalog.get(train_set_name).get("novel_classes", None)
            if not novel_classes:
                raise Exception(
                    f"Dataset {train_set_name} has no novel_classes set, check builtin_meta.py for an example on how to set them")
            novel_txt_files = []
            for class_name in novel_classes:
                for file in txt_files:
                    if os.path.isfile(f"{prev_seed_dir}/{file}") \
                            and class_name in file and f"_{samples_needed}shot" in file:
                        novel_txt_files.append(file)
            if not novel_txt_files:
                raise Exception(f"No novel class txt files found under {prev_seed_dir}")
            # Copy the existing novel class txt files
            for file in novel_txt_files:
                shutil.copy(f"{prev_seed_dir}/{file}", f"{new_seed_dir}")
            # Copy our new base class txt files
            for class_id, file_names in filenames_per_base_class.items():
                # Note: instance filtering is performed later in meta_voc.py
                class_name = self.base_class_id_to_name(class_id)
                with open(f"{new_seed_dir}/box_{samples_needed}shot_{class_name}_train.txt", 'w') as text_file:
                    text_file.write('\n'.join(file_names) + '\n')
        elif 'coco' in train_set_name:
            data_path = "datasets/cocosplit/datasplit/trainvalno5k.json"
            data = json.load(open(data_path))
            logger.info("Loading trainvalno5k.json ...")
            new_all_cats = []
            for cat in data["categories"]:
                new_all_cats.append(cat)

            # Extract relevant images and annotations in a single pass for all classes
            base_classes = MetadataCatalog.get(train_set_name).get("base_classes", None)
            all_filenames = sum(filenames_per_base_class.values(), [])
            all_filenames = [os.path.basename(f) for f in all_filenames]
            id2img = {}
            filename2img = {}
            for img in data["images"]:
                if img["file_name"] in all_filenames:
                    id2img[img["id"]] = img
                    filename2img[img["file_name"]] = img
            all_annotations = []
            for anno in data["annotations"]:
                if anno["image_id"] in id2img.keys() and self.is_base_id(anno["category_id"]) \
                        and not anno.get("is_crowd", 0):
                    all_annotations.append(anno)
            # Match images to their respective classes, annotations to their images, save to json
            new_base_data = {}
            for class_id, file_names in filenames_per_base_class.items():
                file_names = [os.path.basename(f) for f in file_names]
                class_images = [filename2img[filename] for filename in file_names]
                class_image_ids = [img["id"] for img in class_images]
                annotations = [anno for anno in all_annotations if anno["image_id"] in class_image_ids]
                new_base_data[class_id] = {
                    "info": data["info"],
                    "licenses": data["licenses"],
                    "categories": data["categories"],
                    "images": class_images,
                    "annotations": annotations,
                }
            logger.info("Data collated...")
            # Create base files
            for class_id, base_data in new_base_data.items():
                class_name = self.base_class_id_to_name(class_id)
                with open(f"{new_seed_dir}/full_box_{samples_needed}shot_{class_name}_trainval.json",
                          'w') as json_file:
                    json.dump(base_data, json_file)
            # Copy novel files verbatim
            logger.info("Copying novel classes...")
            novel_classes = MetadataCatalog.get(train_set_name).get("novel_classes", None)
            if not novel_classes:
                raise Exception(
                    f"Dataset {train_set_name} has no novel_classes set, check builtin_meta.py for an example on how to set them")
            for class_name in novel_classes:
                source = f"{prev_seed_dir}/full_box_{samples_needed}shot_{class_name}_trainval.json"
                dest = f"{new_seed_dir}/full_box_{samples_needed}shot_{class_name}_trainval.json"
                shutil.copy(source, dest)
                logger.info(f"Copied {class_name}")
        else:
            raise Exception(
                f"You need to implement data split saving for {train_set_name}, check this function and https://github.com/ucbdrive/few-shot-object-detection/tree/master/datasets")
        logger.info(f"Split saved to {new_seed_dir}! ({samples_needed}-shot)")

    def base_class_id_to_name(self, class_id: int):
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if 'voc' in train_set_name:
            base_classes = MetadataCatalog.get(train_set_name).get("base_classes", None)
            return base_classes[class_id]
        elif 'coco' in train_set_name:
            cid_to_contiguous = MetadataCatalog.get(train_set_name).get("base_dataset_id_to_contiguous_id")
            # contiguous_to_cid = {v: k for k, v in cid_to_contiguous.items()}
            return MetadataCatalog.get(train_set_name).get("base_classes")[class_id]
        else:
            raise Exception(
                "You need to specify a class ID mapping for base classes, or add your dataset to this function")

    def is_base_id(self, class_id: int):
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if 'voc' in train_set_name:
            return class_id < 15
        elif 'coco' in train_set_name:
            return class_id in MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get(
                "base_dataset_id_to_contiguous_id").keys()
        else:
            raise Exception(
                "You need to specify a class ID mapping for base classes, or add your dataset to this function")
