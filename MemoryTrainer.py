import re

import torch
from torch import Tensor

from DeFRCNTrainer import DeFRCNTrainer


class MemoryTrainer(DeFRCNTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        memory_config = self.cfg.clone()
        memory_config.defrost()
        # Assuming only 1 dataset at a time, as per usual, e.g. (voc_2007_trainval_novel1_2shot_seed0,)
        train_set_name = memory_config.DATASETS.TRAIN[0]
        name_and_shots, seed = train_set_name.split("_seed")
        new_train_set_name = f"{re.sub('novel_mem|all', 'base_mem', name_and_shots)}_seed{int(seed)}"
        memory_config.DATASETS.TRAIN = [new_train_set_name]
        print(f"Using {new_train_set_name} instead of {train_set_name} for memory")
        # Use same number of shots to be the same as k used in normal config, but different base images
        self.memory_loader = self.build_train_loader(memory_config)
        self._memory_loader_iter = iter(self.memory_loader)
        self.memory_config = memory_config

        # Numerical stability param for A-GEM and CFA, taken from VanDeVen's A-GEM implementation
        self.eps_agem = 1e-7

    def get_memory_batch(self):
        memory_data = next(self._memory_loader_iter)
        # self.adjust_batch_ids(self.memory_config.DATASETS.TRAIN[0], memory_data)
        return memory_data

    def get_current_batch(self):
        current_data = next(self._data_loader_iter)
        # self.adjust_batch_ids(self.cfg.DATASETS.TRAIN[0], current_data)
        return current_data

    def get_gradient(self, model):
        gradient = []
        for p in model.parameters():
            if p.requires_grad:
                gradient.append(p.grad.view(-1))
        return torch.cat(gradient)

    def update_gradient(self, model, new_grad):
        index = 0
        for p in model.parameters():
            if p.requires_grad:
                n_param = p.numel()  # number of parameters in [p]
                p.grad.copy_(new_grad[index:index + n_param].view_as(p))
                index += n_param
