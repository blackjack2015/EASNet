# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

from ofa.stereo_matching.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from ofa.utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock
from ofa.utils import make_divisible, val2list, MyNetwork
from ofa.stereo_matching.networks import AANet

__all__ = ['OFAAANet']


class OFAAANet(AANet):

    def __init__(self, bn_param=(0.1, 1e-5), base_stage_width=None, width_mult=1.0,
                 ks_list=3, expand_ratio_list=6, depth_list=4, scale_list=3):


        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.scale_list = val2list(scale_list, 1)
        self.active_scale = max(self.scale_list)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.scale_list.sort()
        self.feature_blocks = self.ConstructFeatureNet()

        super(OFAAANet, self).__init__(max_disp = 192, feature_blocks = self.feature_blocks, num_scales = max(self.scale_list))
        
        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.fea_block_group_info]

    def ConstructFeatureNet(self):

        base_stage_width = [16*(2**i) for i in range(max(self.scale_list))]

        stride_stages = [3] + [2 for _ in range(max(self.scale_list)-1)]
        act_stages = ['relu' for _ in range(max(self.scale_list))]
        #se_stages = [False, True, False]
        se_stages = [False for _ in range(max(self.scale_list))]
        n_block_list = [max(self.depth_list)] * max(self.scale_list)
        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)

        # inverted residual blocks
        self.fea_block_group_info = []
        blocks = []
        _block_index = 0

        feature_dim = 3
        for width, n_block, s, act_func, use_se in zip(width_list, n_block_list,
                                                       stride_stages, act_stages, se_stages):
            self.fea_block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        return blocks

    def feature_extraction(self, img):
        # blocks
        x = img
        features = []
        for stage_id, block_idx in enumerate(self.fea_block_group_info):
            if stage_id < self.active_scale:
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                for idx in active_idx:
                    x = self.feature_blocks[idx](x)
                features.append(x)
            
        return features

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAAANet'

    @property
    def module_str(self):
        _str = ""

        for stage_id, block_idx in enumerate(self.fea_block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.feature_blocks[idx].module_str + '\n'

        return _str

    @property
    def config(self):
        return {
            'name': OFAAANet.__name__,
            'bn': self.get_bn_param(),
            'fea_blocks': [
                block.config for block in self.feature_blocks
            ],
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    @property
    def grouped_block_index(self):
        return self.fea_block_group_info

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if '.mobile_inverted_conv.' in key:
                new_key = key.replace('.mobile_inverted_conv.', '.conv.')
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif '.bn.bn.' in new_key:
                new_key = new_key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in new_key:
                new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in new_key:
                new_key = new_key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAAANet, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list), s=max(self.scale_list))

    def set_active_subnet(self, ks=None, e=None, d=None, s=None, **kwargs):
        ks = val2list(ks, len(self.feature_blocks))
        expand_ratio = val2list(e, len(self.feature_blocks))
        depth = val2list(d, len(self.fea_block_group_info))

        for block, k, e in zip(self.feature_blocks[:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.fea_block_group_info[i]), d)

        self.active_scale = s

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.feature_blocks))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.feature_blocks))]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.fea_block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):

        blocks = []

        input_channel = 3
        # blocks
        for stage_id, block_idx in enumerate(self.fea_block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.feature_blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.feature_blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = AANet(max_disp = 192, feature_blocks = blocks)

        # make deep copy of other modules
        _subnet.cost_volume = copy.deepcopy(self.cost_volume)
        _subnet.aggregation = copy.deepcopy(self.aggregation)
        _subnet.disparity_estimation = copy.deepcopy(self.disparity_estimation)
        _subnet.refinement = copy.deepcopy(self.refinement)

        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        final_expand_config = self.final_expand_layer.config
        feature_mix_layer_config = self.feature_mix_layer.config
        classifier_config = self.classifier.config

        block_config_list = [first_block_config]
        input_channel = first_block_config['conv']['out_channels']
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append({
                    'name': ResidualBlock.__name__,
                    'conv': self.blocks[idx].conv.get_active_subnet_config(input_channel),
                    'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
                })
                input_channel = self.blocks[idx].conv.active_out_channel
            block_config_list += stage_blocks

        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': first_conv_config,
            'blocks': block_config_list,
            'final_expand_layer': final_expand_config,
            'feature_mix_layer': feature_mix_layer_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.feature_blocks:
            block.conv.re_organize_middle_weights(expand_ratio_stage)
