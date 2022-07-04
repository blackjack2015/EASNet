from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import numpy as np

from ofa.utils import utils
from ofa.utils.file_io import read_img, read_disp


class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform

        sceneflow_finalpass_dict = {
            'train': 'filenames/SceneFlow_finalpass_train.txt',
            'val': 'filenames/SceneFlow_finalpass_val.txt',
            'test': 'filenames/SceneFlow_finalpass_test.txt'
        }

        sintel_dict = {
            'train': 'filenames/Sintel_ALL.txt',
            'val': 'filenames/Sintel_ALL.txt',
            'test': 'filenames/Sintel_ALL.txt'
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_2015_test = {
            'train': 'filenames/KITTI_2015_test.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_2012_test = {
            'train': 'filenames/KITTI_2012_test.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_sf_mix_dict = {
            'train': 'filenames/sf_kitti_mix.txt',
            'test': 'filenames/KITTI_mix.txt'
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_MIX': kitti_mix_dict,
            'KITTI_SF_MIX': kitti_sf_mix_dict,
            'KITTI_2015_TEST': kitti_2015_test,
            'KITTI_2012_TEST': kitti_2012_test,
            'Sintel': sintel_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]

            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None

            if load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')
                # KITTI 2012
                elif 'disp_occ' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                                     'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]

        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        sample['disp_name'] = sample_path['left']

        # padding for KITTI
        if self.dataset_name in ['KITTI2012', 'KITTI2015', 'KITTI_MIX', 'KITTI_SF_MIX']:
            h, w, _ = sample['left'].shape
            top_pad = 544-h
            left_pad = 1296-w
            sample['left'] = np.lib.pad(sample['left'],((top_pad,0),(left_pad,0),(0,0)),mode='constant',constant_values=0)
            sample['right'] = np.lib.pad(sample['right'],((top_pad,0),(left_pad,0),(0,0)),mode='constant',constant_values=0)
            sample['disp'] = np.lib.pad(sample['disp'],((top_pad,0),(left_pad,0)),mode='constant',constant_values=0)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
