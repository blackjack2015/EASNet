# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import torch
import argparse

from ofa.stereo_matching.data_providers.stereo import StereoDataProvider
from ofa.stereo_matching.run_manager import StereoRunConfig, RunManager
from ofa.stereo_matching.elastic_nn.networks.ofa_aanet import OFAAANet
from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    help='The path of stereo dataset',
    type=str,
    default='/home/datasets/SceneFlow')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='0')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=1)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=4)
parser.add_argument(
    '-n',
    '--net',
    metavar='OFAAANet',
    default='ofa_aanet',
    choices=['ofa_aanet_d234_e346_k357_w1.0',
             'ofa_aanet'],
    help='OFA AANet networks')

args = parser.parse_args()
if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
StereoDataProvider.DEFAULT_PATH = args.path

ofa_network = OFAAANet(ks_list=[3,5,7], expand_ratio_list=[4,5,6,8], depth_list=[2,3,4], scale_list=[2,3,4])
run_config = StereoRunConfig(test_batch_size=args.batch_size, n_worker=args.workers)

model_file = 'ofa_stereo_checkpoints/ofa_stereo_D4_E8_K7_S4'
init = torch.load(model_file, map_location='cpu')
model_dict = init['state_dict']
ofa_network.load_state_dict(model_dict)

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
#ofa_network.sample_active_subnet()
#ofa_network.set_max_net()
ofa_network.set_active_subnet(ks=3, d=2, e=2, s=2)
subnet = ofa_network.get_active_subnet(preserve_weight=True)
#subnet = ofa_network

""" Test sampled subnet 
"""
run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)
# assign image size: 128, 132, ..., 224
#run_config.data_provider.assign_active_img_size(224)
run_manager.reset_running_statistics(net=subnet)

print('Test random subnet:')
print(subnet.module_str)

loss, (epe, d1, thres1, thres2, thres3) = run_manager.validate(net=subnet)
print('Results: loss=%.5f,\t epe=%.2f' % (loss, epe))
