
# Determinism
import logging
import os

import torch
from detectron2.engine import launch

from BaseAblationSampler import BaseAblationSampler
from BaseProtoSampler import BaseProtoSampler
from defrcn.config import get_cfg, set_global_cfg
from defrcn.engine import default_setup, default_argument_parser

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if "1.8.2" in torch.__version__:
    torch.use_deterministic_algorithms(True)
else:
    torch.use_deterministic_algorithms(True, warn_only=True)

logger = logging.getLogger(__name__)


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    # Due to bug in yacs' merge_from_list (does not respect set_new_allowed when input is list)
    if args.opts:
        cfg.merge_from_list(args.opts)
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # This is specifically for base classes
    assert "base" in cfg.DATASETS.TRAIN[0]
    # Args
    ablation = bool(args.ablation)
    prev_seed = args.prev_seed
    new_seed = args.new_seed
    samples_needed = args.sample_out_size

    if ablation:
        sampler = BaseProtoSampler(cfg)
    else:
        sampler = BaseAblationSampler(cfg)
    prototypes = sampler.build_prototypes(args.sample_pool_size)
    for s in samples_needed:
        filenames_per_class = sampler.filter_samples(prototypes, s)
        sampler.save(filenames_per_class, s, prev_seed, new_seed)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--ablation', dest='ablation', default=0, type=int, help="Use random sampling instead of herd")
    parser.add_argument('--sample_pool_size', dest='sample_pool_size', default=100, type=int,
                        help="Number of samples to draw prototypes from")
    parser.add_argument('--sample_out_size', dest='sample_out_size', default=5, nargs="+", type=int,
                        help="Number of shots needed as output")
    parser.add_argument('--prev_dataseed', dest='prev_seed', default=0, type=int,
                        help="Int ID of one of your existing splits. This is required for keeping novel classes intact")
    parser.add_argument('--new_dataseed', dest='new_seed', type=int,
                        help="Int ID of output split, e.g. 30. This will add a new split directory or replace an existing one")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
