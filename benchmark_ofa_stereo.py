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
import numpy as np
from ofa.utils.pytorch_utils import get_net_info

parser = argparse.ArgumentParser()
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='0')
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

ofa_network = OFAAANet(ks_list=[3,5,7], expand_ratio_list=[2,4,6,8], depth_list=[2,3,4], scale_list=[2,3,4])

model_file = 'ofa_stereo_checkpoints/ofa_stereo_D234_E2468_K357_S4'
init = torch.load(model_file, map_location='cpu')
model_dict = init['state_dict']
ofa_network.load_state_dict(model_dict)

""" Randomly sample a sub-network, 
    you can also manually set the sub-network using: 
        ofa_network.set_active_subnet(ks=7, e=6, d=4) 
"""
#ofa_network.sample_active_subnet()
#ofa_network.set_max_net()
ks = 3
d = 2
e = 2
s = 2
ofa_network.set_active_subnet(ks=ks, d=d, e=e, s=s)
subnet = ofa_network.get_active_subnet(preserve_weight=True)
#subnet = ofa_network
save_path = "ofa_stereo_checkpoints/aanet_D%d_E%d_K%d_S%d" % (d, e, ks, s)
torch.save(subnet.state_dict(), save_path)

net = subnet
net.eval()
net = net.cuda()
#net = net.get_tensorrt_model()
#torch.save(net.state_dict(), 'models/mobilefadnet_trt.pth')
get_net_info(net, input_shape=(3, 540, 960))

# fake input data
dummy_left = torch.randn(1, 3, 576, 960, dtype=torch.float).cuda()
dummy_right = torch.randn(1, 3, 576, 960, dtype=torch.float).cuda()

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 30
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _ = net(dummy_left, dummy_right)
    # MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(-3, repetitions):
        starter.record()
        _ = net(dummy_left, dummy_right)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        if rep >= 0:
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            print(rep, curr_time)

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)


