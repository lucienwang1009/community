FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04@sha256:97cb0ab968d90e0c828d402d277a0086e8a7db3c00301efcd3fcb59fef78cc6d
# CUDA: 9.0.176, CUDNN: 7.2.1.38


WORKDIR /research

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip


ENV HOME /research
ENV PYTHON_VER python3.5
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

RUN apt-get install -y python-setuptools python-pip python3-pip virtualenv htop python3-tk libgtk2.0
RUN virtualenv -p $PYTHON_VER $PYENV_ROOT
RUN pip3 install --upgrade numpy scipy sklearn tensorflow-gpu==1.9.0

# Mount data into the docker
ADD . /research/object_detection

WORKDIR /research/object_detection
RUN pip3 install -r ./requirements.txt
<<<<<<< HEAD
=======
RUN python3 setup.py install
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh

# Fetch python api for coco dataset
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi && \
    cd /cocoapi && \
    git checkout ed842bffd41f6ff38707c4f0968d2cfd91088688
<<<<<<< HEAD
RUN cd /cocoapi/PythonAPI && make
RUN cp -r /cocoapi/PythonAPI/pycocotools /research/object_detection

=======
WORKDIR /cocoapi/PythonAPI
RUN make
RUN cp -r /cocoapi/PythonAPI/pycocotools /research/object_detection

WORKDIR /research/object_detection

>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
ENTRYPOINT ["/bin/bash"]

