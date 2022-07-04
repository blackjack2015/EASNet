from __future__ import print_function
import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F
import skimage
import skimage.io
import numpy as np
import time
import math
from ofa.stereo_matching.data_providers import transforms

from ofa.stereo_matching.run_manager import StereoRunConfig, RunManager
from ofa.stereo_matching.elastic_nn.networks.ofa_aanet import OFAAANet
from ofa.stereo_matching.elastic_nn.utils import set_running_statistics
from ofa.stereo_matching.elastic_nn.training.progressive_shrinking import load_models

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='FADNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--savepath', default='results/',
                    help='path to save the results.')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.datapath)

devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

fullnet = OFAAANet(ks_list=[3,5,7], expand_ratio_list=[2,4,6,8], depth_list=[2,3,4], scale_list=[2,3,4])

model_file = args.loadmodel
init = torch.load(model_file, map_location='cpu')
model_dict = init['state_dict']
fullnet.load_state_dict(model_dict)

d = 2
e = 8
ks = 7
s = 4
fullnet.set_active_subnet(ks=ks, d=d, e=e, s=s)
model = fullnet.get_active_subnet(preserve_weight=True)

# set the batch norm values with testing data
from ofa.stereo_matching.data_providers.stereo import StereoDataProvider
if args.KITTI == '2015':
    StereoDataProvider.DEFAULT_PATH = '/datasets/kitti2015/'
    dataname='KITTI_2015_TEST'
else:
    StereoDataProvider.DEFAULT_PATH = '/datasets/kitti2012/'
    dataname='KITTI_2012_TEST'
run_config = StereoRunConfig(test_batch_size=4, n_worker=4, dataname=dataname)
run_manager = RunManager('.tmp/eval_subnet', model, run_config, init=False)
run_manager.reset_running_statistics(net=model, subset_size=200, subset_batch_size=16)

model = nn.DataParallel(model, device_ids=devices)
model.cuda()

def test(imgL,imgR):
    model.eval()

    if args.cuda:
       imgL = imgL.cuda()
       imgR = imgR.cuda()     

    #print(imgL.size(), imgR.size())
    with torch.no_grad():
        output = model(imgL, imgR)[-1]
        output = torch.squeeze(output)

    pred_disp = output.data.cpu().numpy()

    print(pred_disp.shape)
    #print('larger than 192: %s' % pred_disp[pred_disp>0.75].shape)
    print('min: %f, max: %f, mean: %f' % (np.min(pred_disp), np.max(pred_disp), np.mean(pred_disp)))

    return pred_disp

def main():

    for inx in range(len(test_left_img)):
        print('image: %s'%test_left_img[inx])

        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))

        imgs = {'left':imgL_o, 'right':imgR_o}
        val_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        val_transforms = transforms.Compose(val_transform_list)
        rgb_transform = val_transforms
        imgs = rgb_transform(imgs)
        imgL = imgs['left'].unsqueeze(0)
        imgR = imgs['right'].unsqueeze(0)

        # pad to resize (384, 1280)
        top_pad = 384-imgL.shape[2]
        right_pad = 1280-imgL.shape[3]
        # imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,right_pad)),mode='constant',constant_values=0)
        # imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,right_pad)),mode='constant',constant_values=0)
        imgL = F.pad(imgL,(0, right_pad, top_pad, 0),mode='constant',value=0)
        imgR = F.pad(imgR,(0, right_pad, top_pad, 0),mode='constant',value=0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))

        img = pred_disp[top_pad:,:-right_pad]

        round_img = np.round(img*256)

        skimage.io.imsave(os.path.join(args.savepath, test_left_img[inx].split('/')[-1]),round_img.astype('uint16'))

if __name__ == '__main__':
   main()






