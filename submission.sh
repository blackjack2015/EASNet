PY="/home/esetstore/blackjack/anaconda3/bin/python"
VER=2015
model_path=./ofa_stereo_checkpoints/kitti${VER}_pre
save_path=./submit_results/easnet-m-kitti${VER}/
$PY kitti_submission.py --maxdisp 192 \
                     --KITTI ${VER} \
                     --datapath /datasets/kitti${VER}/testing/ \
                     --savepath $save_path \
                     --loadmodel $model_path \
