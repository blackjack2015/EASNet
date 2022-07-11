#!/usr/bin/env bash

# esetstore MPI setting
PY="python"
MPIPATH="/usr"
hosts="-np 8 -H host1:4,host2:4"

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$PY train_ofa_stereo.py --task kitti2015
