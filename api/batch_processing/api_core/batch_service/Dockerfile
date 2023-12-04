FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# Python version is 3.6.8
RUN apt-get update && apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    git \
    libxext6 \
    libglib2.0-0 \
    libgl1 

RUN git clone https://github.com/ultralytics/yolov5/

RUN pip install ultralytics-yolov5 yolov5 ultralytics
RUN pip install --upgrade pip
RUN pip install azure-storage-blob==12.7.1 pillow numpy requests jsonpickle
