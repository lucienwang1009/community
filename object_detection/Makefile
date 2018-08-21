SELECTED_SUBSCRIPTION="GPU Deep Learning Compute"
LOCATION:=eastus
GROUP_NAME:=MLPerfEastUS
STORAGE_ACCOUNT_NAME:=mlperfstorage
CLUSTER_NAME:=mlperf
WORKSPACE:=mlperf

BENCHMARK_NAME:=object_detection
JOB_NAME:=mlperf_$(BENCHMARK_NAME)_tfr1_9
EXPERIMENT:=$(BENCHMARK_NAME)_tfr1_9
IMAGE_NAME:="mlperfregistry.azurecr.io/mlperf/$(BENCHMARK_NAME):cuda9-cudnn7-tf1.9.0"
DOCKER_DIR:=tensorflow

include ../../include/control.mk
