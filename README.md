# EASNet
EASNet: Searching Elastic and Accurate Network Architecture for Stereo Matching (ECCV 2022)
[[arXiv]]() [[Video]]()
PyTorch implementation of searching an efficient network architecture for stereo matching. Please cite the paper below if you use this project. Any suggestion, fork, and pull request is welcome. 
```BibTex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```

## TL;DR quickstart

## Setup

Python 3 dependencies:

* PyTorch 1.8+
* OpenMPI 4.0.1+
* matplotlib
* numpy
* imageio
* other necessary packages

We use the deformation module from AANet. Install the ``deform_conv'' package as follow.
```
cd ofa/stereo_matching/networks/deform_conv
sh build.sh
```

Our training scripts apply MPI to accelerate the training procedure. Please install OpenMPI 4.0.1 or above. 

## Searching EASNet
The main commands are summarized in ``train.sh''. One can use them accordingly. 

### Train the largest supernet.
```
# two nodes, four GPUs per node
mpirun -np 8 -H host1:4,host2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_stere.py \
           --task large
```
### Shrink the kernel size/depth/width/scale.
```
# two nodes, four GPUs per node
export TASK=kernel  # 'kernel', 'depth', 'width', 'scale'
mpirun -np 8 -H host1:4,host2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_stere.py \
           --task $TASK
```

## Finetuning on KITTI 2012/2015
```
# two nodes, four GPUs per node
export TASK=kitti12  # 'kitti12', 'kitti2015'
mpirun -np 8 -H host1:4,host2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train_ofa_stere.py \
           --task $TASK
```

## Pretrained Models
