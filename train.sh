#!/usr/bin/env bash

# esetstore MPI setting
PY="python"
MPIPATH="/usr"
hosts="-np 8 -H host1:4,host2:4"

# train the large super net
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$PY train_ofa_stereo.py --task large

# shrink the kernel size
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$PY train_ofa_stereo.py --task kernel

# shrink the network depth
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$PY train_ofa_stereo.py --task depth

# shrink the network width
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$PY train_ofa_stereo.py --task width

# shrink the network scale
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH $hosts -bind-to none -map-by slot \
	$PY train_ofa_stereo.py --task scale

