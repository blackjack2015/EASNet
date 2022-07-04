#!/usr/bin/env bash

# slurm batch script
#SBATCH -o /home/comp/qiangwang/blackjack/once-for-all/normal_v2.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -w hkbugpusrv01,hkbugpusrv03

## DAAI MPI setting
#PY="/usr/local/bin/python"
#MPIPATH="/home/comp/qiangwang/software/openmpi-4.0.1"
#params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1  \
#	--mca btl_tcp_if_include ib0  \
#	--mca btl_openib_want_fork_support 1   \
#	-x LD_LIBRARY_PATH    \
#	-x NCCL_IB_DISABLE=0  \
#	-x NCCL_SOCKET_IFNAME=ib0  \
#	-x NCCL_DEBUG=INFO  \
#	-x HOROVOD_CACHE_CAPACITY=0"
##hosts="-np 4 -H hkbugpusrv03:4"
#hosts="-np 8 -H hkbugpusrv01:4,hkbugpusrv03:4"

# esetstore MPI setting
PY="/home/esetstore/blackjack/anaconda3/bin/python"
MPIPATH="/home/esetstore/local/openmpi-4.0.1"
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1  \
	--mca btl_tcp_if_include ib0  \
	--mca btl_openib_want_fork_support 1   \
	-x LD_LIBRARY_PATH    \
	-x NCCL_IB_DISABLE=0  \
	-x NCCL_SOCKET_IFNAME=ib0  \
	-x NCCL_DEBUG=INFO  \
	-x HOROVOD_CACHE_CAPACITY=0"
#hosts="-np 4 -H gpu16:4"
hosts="-np 16 -H gpu1:4,gpu2:4,gpu3:4,gpu4:4"
#hosts="-np 1 -H gpu16:1"
#hosts="-np 16 -H gpu6:4,gpu7:4,gpu15:4,gpu16:4"
#hosts="-np 16 -H gpu7:4,gpu11:4,gpu12:4,gpu16:4"

# train the large super net
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
#	$params \
#	$PY train_ofa_stereo.py --task large

# shrink the kernel,depth,width
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$params \
	$PY train_ofa_stereo.py --task kitti2015
