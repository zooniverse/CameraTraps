FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Python version is 3.10.12
RUN apt-get update && apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    python3 \
    python3-pip \
    git \
    libxext6 \
    libglib2.0-0 \
    libgl1

RUN git clone https://github.com/ultralytics/yolov5/

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install ultralytics-yolov5 yolov5 ultralytics

RUN pip install azure-storage-blob==12.7.1 pillow numpy requests jsonpickle