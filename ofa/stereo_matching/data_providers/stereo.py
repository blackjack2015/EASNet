# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ofa.stereo_matching.data_providers import transforms
from .base_provider import DataProvider
from .dataset import StereoDataset
from ofa.utils.my_dataloader import MyDistributedSampler

__all__ = ['StereoDataProvider']


class StereoDataProvider(DataProvider):
    DEFAULT_PATH = '/datasets/SceneFlow'

    def __init__(self, save_path=None, train_batch_size=16, test_batch_size=32, valid_size=None, n_worker=8,
                 dataset_name='SceneFlow',load_pseudo_gt=False,
                 num_replicas=None, rank=None):

        warnings.filterwarnings('ignore')
        self._save_path = save_path
        self._dataset_name = dataset_name
        self._load_pseudo_gt = load_pseudo_gt

        self._img_height = 540
        self._img_width = 960
        self._crop_height = 384
        self._crop_width = 768

        train_dataset = self.train_dataset(self.build_train_transform())

        valid_transforms = self.build_valid_transform()
        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset), valid_size)

            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, True, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, True, np.array(valid_indexes))
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                self.train = torch.utils.data.DataLoader(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True
                )
            else:
                self.train = torch.utils.data.DataLoader(
                    train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
                )
            self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'stereo'

    @property
    def data_shape(self):
        return [[3, self._img_height, self._img_width], [3, self._img_height, self._img_width]]  # C, H, W

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser('~/dataset/imagenet')
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        return StereoDataset(self.save_path, mode='train', transform=_transforms)

    def test_dataset(self, _transforms):
        return StereoDataset(self.save_path, mode='test', transform=_transforms)

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, print_log=True):
        train_transform_list = [transforms.RandomCrop(self._crop_height, self._crop_width),
                                transforms.RandomColor(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]

        train_transforms = transforms.Compose(train_transform_list)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        val_transform_list = [
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ]
        val_transforms = transforms.Compose(val_transform_list)
        return val_transforms

    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting BN running statistics
        #if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
        if self.__dict__.get('sub_train_list', None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, True, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_list'] = []
            for sample in sub_data_loader:
                self.__dict__['sub_train_list'].append(sample)
        #return self.__dict__['sub_train_%d' % self.active_img_size]
        return self.__dict__['sub_train_list']
