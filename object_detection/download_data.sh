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

ln -s cocoapi/PythonAPI/pycocotools .
python3 data_download.py --dataset ./coco_dataset --year 2014
unlink pycocotools

# download weights pre-trained by imagenet
mkdir weights
pushd weights
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
popd
