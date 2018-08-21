#/bin/bash

RANDOM_SEED=$1
QUALITY=$2
set -e

export PYTHONPATH=`pwd`:$PYTHONPATH

<<<<<<< HEAD
python3 samples/coco/coco.py train --seed=$RANDOM_SEED --dataset=/coco_dataset --model=/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --logs=/tmp/rcnn_logs --year=2014
=======
python3 samples/coco/coco.py train --seed=$RANDOM_SEED --dataset=/coco_dataset --model=/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --logs=/tmp/rcnn_logs --year=2014 --min_maskap=$QUALITY
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
