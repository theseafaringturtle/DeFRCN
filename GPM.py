import logging
import time
from abc import ABC
from collections import OrderedDict
from functools import partial
from typing import Dict, Tuple
import torch.nn.functional as F

import numpy as np
import torch
from detectron2.utils.comm import get_rank
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger("defrcn").getChild(__name__)



def compute_conv_output_size(input_size: Tuple[int, int], kernel_size: Tuple[int, int],
                             stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    conv_out_size = []
    for dim in range(2):
        size_dim = int(np.floor((input_size[dim] + 2 * padding[dim] - dilation[dim] * (kernel_size[dim] - 1) - 1)))
        conv_out_size.append(size_dim)
    return conv_out_size


def register_feature_map_hooks(model: nn.Module) -> [RemovableHandle]:
    def capture_activation(layer_name: str, module: nn.Module, input: Tensor):
        model.fmap.set_activation_for_layer(layer_name, input[0])

    hook_handles = []
    for name, module in model.named_modules():
        if name in model.fmap.layer_names:
            handle = module.register_forward_pre_hook(partial(capture_activation, name))
            hook_handles.append(handle)
            model.fmap.set_module(name, module)
    model.fmap.activations_hooks_registered = True
    return hook_handles


def determine_conv_output_sizes(model: nn.Module, random_samples):
    """Run a single sample through network, get output size of all the convolutions we've marked.
    This is useful in case we don't just pick sequential convolution in the network
    :param: model: full model, with layer_names already marked in model.fmap
    :param: random_samples: data in the format the model expects. Can be a tensor of tensors (classification) or a list (detectron2).
    random_samples should use the maximum input image size to the network, including RGB channels
    """
    assert hasattr(model, 'fmap'), "Model needs feature map storage"
    assert model.fmap.activations_hooks_registered is True, "Need to register hooks to sizes of convolutional layers across the model"
    out_sizes: Dict[str, (int, int)] = {}

    model.fmap.clear_activations()
    model.eval()
    with torch.no_grad():
        # According to pytorch, mini-batch stats are used in training mode, and in eval mode when buffers are None.
        # since we're not tracking stats as per GPM, we need at least 2 samples
        model(random_samples)
    for layer_name in model.fmap.layer_names:
        output_size = model.fmap.get_activation_for_layer(layer_name).shape
        if model.fmap.is_conv_layer(layer_name):
            bsize, channels, w, h = output_size
            out_sizes[layer_name] = (w, h)
        else:
            bsize, linear_out = output_size
            out_sizes[layer_name] = linear_out
    model.fmap.clear_activations()
    # min_size currently unused
    for layer_name in model.fmap.layer_names:
        model.fmap.set_input_size(layer_name, out_sizes[layer_name])


class FeatureMap(ABC):
    def __init__(self):
        self.layer_names = ['your', 'layers', 'here']
        self.samples = {
            'your': 16,
            'layers': 16,
            'here': 16
        }
        self.final_proj_samples = 16
        self.modules = OrderedDict()
        self.act = OrderedDict()
        self.input_sizes = OrderedDict()
        self.activations_hooks_registered = False
        self.getting_conv_size = True

    def get_samples(self, module_name: str):
        return self.samples[module_name]

    def set_module(self, module_name: str, module: nn.Module):
        assert module_name in self.layer_names, f"Module {module_name} not in layers, allowed: {list(self.layer_names)}"
        self.modules[module_name] = module

    def get_module(self, module_name: str):
        return self.modules[module_name]

    def set_activation_for_layer(self, module_name: str, input: Tensor):
        assert module_name in self.layer_names, f"Module {module_name} not in layers, allowed: {list(self.layer_names)}"
        if 0 in input.shape:  # Don't mess activation map with 0-dim outputs
            raise Exception(
                f"Empty dimension in {module_name}, if hooking roi_heads make sure valid proposals have been fed through")
        else:
            # Quick hack to distinguish between actual calculation and initial pass to get feature map sizes
            if self.getting_conv_size:
                self.act[module_name] = input
            else:
                conv_width, conv_height = compute_conv_output_size(self.get_input_size(module_name),
                                                                   self.get_kernel_size(module_name))
                # perform average pooling for larger inputs
                act = F.adaptive_avg_pool2d(input, (conv_width, conv_height))
                if module_name not in self.act or self.act[module_name] is None:
                    self.act[module_name] = act
                else:
                    self.act[module_name] = torch.cat((self.act[module_name], act), dim=0)

    def get_activation_for_layer(self, module_name: str):
        return self.act[module_name]

    def clear_activations(self):
        self.act = OrderedDict()

    def get_all_activations(self) -> [str, Tensor]:
        return list(self.act.items())

    def set_input_size(self, layer_name, size: Tuple[int, int]):
        # Could be max (0-pad smaller inputs in that case) or min (perform average pooling on larger inputs)
        self.input_sizes[layer_name] = size

    def get_input_size(self, layer_name):
        return self.input_sizes[layer_name]

    def is_conv_layer(self, module_name: str):
        return hasattr(self.get_module(module_name), 'kernel_size')

    def get_kernel_size(self, module_name: str):
        return self.get_module(module_name).kernel_size

    def get_in_channel(self, module_name: str):
        module = self.get_module(module_name)
        # Could add assert here
        return module.in_channels


def get_representation_matrix(net, device) -> Dict[str, Tensor]:
    logger.info(f"Computing representation matrix")
    clock_start_comp = time.perf_counter()
    # Get representation matrix (note: largest input size was used as baseline for convolutions in determine_conv_output_sizes)
    mats = dict()
    for layer_name in net.fmap.layer_names:
        batch_size = net.fmap.samples.get(layer_name)
        k = 0
        if net.fmap.is_conv_layer(layer_name):
            kernel_size = net.fmap.get_kernel_size(layer_name)
            in_channel = net.fmap.get_in_channel(layer_name)
            conv_width, conv_height = compute_conv_output_size(net.fmap.get_input_size(layer_name),
                                                               net.fmap.get_kernel_size(layer_name))
            mat = torch.zeros((kernel_size[0] * kernel_size[1] * in_channel, conv_width * conv_height * batch_size))
            act = net.fmap.get_activation_for_layer(layer_name).detach()
            for kk in range(batch_size):
                for w_index in range(conv_width - kernel_size[0]):
                    for h_index in range(conv_height - kernel_size[1]):
                        patch = act[kk, :, w_index:kernel_size[0] + w_index, h_index:kernel_size[1] + h_index]
                        # Got rid of 0-padding code, since I'm performing average pooling for minimum anyway
                        mat[:, k] = patch.reshape(-1)
                        k += 1
            mats[layer_name] = mat.to(device)
        else:
            act = net.fmap.get_activation_for_layer(layer_name).detach()
            activation = act[0:batch_size].transpose(1, 0)
            mats[layer_name] = activation.to(device)
    clock_end_comp = time.perf_counter()
    logger.debug(f"Representation matrix computation time: {clock_end_comp - clock_start_comp}")
    return mats


def update_GPM(mat_dict, threshold, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if not features:
        # After First Task
        clock_start = time.perf_counter()
        for layer_name in mat_dict.keys():
            activation = mat_dict[layer_name]
            U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = torch.sum(torch.cumsum(sval_ratio.reshape(-1), dim=0) < threshold[layer_name])  # +1
            features[layer_name] = U[:, 0:r]

            clock_end = time.perf_counter()
            if get_rank() == 0:  # Don't print for every GPU process
                logger.debug(f"SVD time for {layer_name}: {clock_end - clock_start}")
            clock_start = clock_end
    else:  # Not currently in use, given G-FSOD setting
        for layer_name in mat_dict.keys():
            activation = mat_dict[layer_name]
            U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - torch.matmul(
                torch.matmul(features[layer_name], features[layer_name].transpose(1, 0)), activation)
            U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[layer_name]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                logger.debug('Skip Updating GPM for layer: {}'.format(layer_name))
                continue
            # update GPM
            Ui = torch.hstack((features[layer_name], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                features[layer_name] = Ui[:, 0:Ui.shape[0]]
            else:
                features[layer_name] = Ui

    if get_rank() == 0:  # Don't print for every GPU process
        logger.debug('-' * 40)
        logger.debug('Gradient Constraints Summary')
        logger.debug('-' * 40)
        for layer_name in features:
            logger.debug(
                'Layer {} : {}/{}'.format(layer_name, features[layer_name].shape[1], features[layer_name].shape[0]))
        logger.debug('-' * 40)
    return features
