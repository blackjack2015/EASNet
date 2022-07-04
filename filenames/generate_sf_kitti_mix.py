import random

sf_list = []
kitti_list = []
mix_list = []

with open('SceneFlow_finalpass_train.txt', 'r') as f:
    sf_list.extend(f.readlines())

with open('KITTI_mix.txt', 'r') as f:
    kitti_list.extend(f.readlines())

ratio = 0.3
sf_sample_number = int(len(kitti_list) * ratio)

random.shuffle(sf_list)
mix_list.extend(sf_list[:sf_sample_number])
mix_list.extend(kitti_list)

with open('sf_kitti_mix.txt', 'w') as f:
    f.writelines(mix_list)

