# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random

import horovod.torch as hvd
import torch

from ofa.stereo_matching.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from ofa.stereo_matching.elastic_nn.networks import OFAAANet
from ofa.stereo_matching.run_manager import DistributedStereoRunConfig
from ofa.stereo_matching.run_manager.distributed_run_manager import DistributedRunManager
from ofa.utils import download_url, MyRandomResizedCrop
from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import load_models


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='large', choices=[
    'kernel',
    'depth',
    'expand',
    'scale',
    'large',
    'final',
    'kitti2012',
    'kitti2015',
    'kitti_mix',
    'kitti_sf_mix',
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()

args.manual_seed = 0
args.dataname = 'SceneFlow'

args.lr_schedule_type = 'cosine'

args.base_batch_size = 2
args.valid_size = None

args.opt_type = 'adam'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 1e-4
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = 'tf'
args.image_size = '128,160,192,224'
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = 'ce'

if args.task == 'large':
    args.path = 'exp/normal'
    args.dynamic_batch_size = 1
    args.n_epochs = 64
    args.base_lr = 1e-3
    args.warmup_epochs = 0
    args.warmup_lr = -1
    args.ks_list = '7'
    args.expand_list = '8'
    args.depth_list = '4'
    args.scale_list = '4'
elif args.task == 'kernel':
    args.path = 'exp/normal2kernel'
    args.dynamic_batch_size = 1
    args.n_epochs = 25
    args.base_lr = 5e-4
    args.warmup_epochs = 2
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '8'
    args.depth_list = '4'
    args.scale_list = '4'
elif args.task == 'depth':
    args.path = 'exp/kernel2kernel_depth'
    args.dynamic_batch_size = 2
    args.n_epochs = 25
    args.base_lr = 5e-4
    args.warmup_epochs = 2
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '8'
    args.depth_list = '2,3,4'
    args.scale_list = '4'
elif args.task == 'expand':
    args.path = 'exp/kernel_depth2kernel_depth_width'
    args.dynamic_batch_size = 4
    args.n_epochs = 25
    args.base_lr = 5e-4
    args.warmup_epochs = 2
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '2,4,6,8'
    args.depth_list = '2,3,4'
    args.scale_list = '4'
elif args.task == 'scale':
    args.path = 'exp/kernel_depth_width2kernel_depth_width_scale'
    args.dynamic_batch_size = 6
    args.n_epochs = 25
    args.base_lr = 5e-4
    args.warmup_epochs = 2
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '2,4,6,8'
    args.depth_list = '2,3,4'
    args.scale_list = '2,3,4'
elif args.task in ['kitti2012', 'kitti2015', 'kitti_mix', 'kitti_sf_mix']: # finetune on kitti
    args.path = 'exp/%s' % args.task
    args.dynamic_batch_size = 6
    args.n_epochs = 400    # since the dynamic batch size leads to multiple step.
    args.warmup_epochs = 0
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '2,4,6,8'
    args.depth_list = '2,3,4'
    args.scale_list = '2,3,4'
    args.datapath = '/datasets/'
    if args.task == 'kitti2012':
        args.datapath = '/datasets/kitti2012'
    if args.task == 'kitti2015':
        args.datapath = '/datasets/kitti2015'
    args.dataname = args.task.upper()
    from ofa.stereo_matching.data_providers.stereo import StereoDataProvider
    StereoDataProvider.DEFAULT_PATH = args.datapath
    args.lr_schedule_type = 'multistep-100-0.5'
    args.base_lr = 1e-4
else:
    raise NotImplementedError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    os.makedirs(args.path, exist_ok=True)

    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    #args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    args.init_lr = args.base_lr
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedStereoRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]
    args.scale_list = [int(d) for d in args.scale_list.split(',')]

    args.width_mult_list = args.width_mult_list[0] if len(args.width_mult_list) == 1 else args.width_mult_list
    net = OFAAANet(
        bn_param=(args.bn_momentum, args.bn_eps),
        base_stage_width=args.base_stage_width, width_mult=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list, scale_list=args.scale_list
    )

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()
    #print('Finish broadcasting.')

    # training
    from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import validate, train

    validate_func_dict = {'image_size_list': {224},
                          'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          'depth_list': sorted({min(net.depth_list), max(net.depth_list)}),
                          'scale_list': sorted({min(net.scale_list), max(net.scale_list)})}
    if args.task == 'large':
        train(distributed_run_manager, args,
              lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    elif args.task == 'kernel':
        validate_func_dict['ks_list'] = sorted(args.ks_list)
        if distributed_run_manager.start_epoch == 0:
            args.ofa_checkpoint_path = 'ofa_stereo_checkpoints/ofa_stereo_D4_E8_K7_S4'
            load_models(distributed_run_manager, distributed_run_manager.net, args.ofa_checkpoint_path)
        else:
            assert args.resume
        train(distributed_run_manager, args,
              lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    elif args.task == 'depth':
        from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import train_elastic_depth
        args.ofa_checkpoint_path = 'ofa_stereo_checkpoints/ofa_stereo_D4_E8_K357_S4'
        train_elastic_depth(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == 'expand':
        from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import train_elastic_expand
        args.ofa_checkpoint_path = 'ofa_stereo_checkpoints/ofa_stereo_D234_E8_K357_S4'
        train_elastic_expand(train, distributed_run_manager, args, validate_func_dict)
    elif args.task == 'scale':
        from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import train_elastic_scale
        args.ofa_checkpoint_path = 'ofa_stereo_checkpoints/ofa_stereo_D234_E2468_K357_S4'
        train_elastic_scale(train, distributed_run_manager, args, validate_func_dict)
    elif (args.task in ['kitti2012', 'kitti2015', 'kitti_mix', 'kitti_sf_mix']):
        from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import train_elastic_scale
        if args.task in ['kitti2012', 'kitti2015']:
            args.ofa_checkpoint_path = 'ofa_stereo_checkpoints/kitti_sf_mix'
        else:
            args.ofa_checkpoint_path = 'ofa_stereo_checkpoints/ofa_stereo_D234_E2468_K357_S234'
        train_elastic_scale(train, distributed_run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError
