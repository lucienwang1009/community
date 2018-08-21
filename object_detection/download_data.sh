#!/bin/bash

if [[ ! -d cocoapi ]]; then
  git clone https://github.com/cocodataset/cocoapi.git
  pushd cocoapi
  git checkout ed842bffd41f6ff38707c4f0968d2cfd91088688
  popd
fi

pushd cocoapi/PythonAPI
make
popd

<<<<<<< HEAD
if [[ ! -d pycocotools ]]; then
  ln -s cocoapi/PythonAPI/pycocotools/ .
fi

python3 data_download.py --dataset ./coco_dataset --year 2014

# download weights pre-trained by imagenet
mkdir weights
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 ./weights
=======
ln -s cocoapi/PythonAPI/pycocotools .
python3 data_download.py --dataset ./coco_dataset --year 2014
unlink pycocotools

# download weights pre-trained by imagenet
mkdir weights
pushd weights
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
popd
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
