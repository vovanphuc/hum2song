FROM nvcr.io/nvidia/pytorch:21.07-py3
USER root

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install ffmpeg libavcodec-extra -y && \
    apt-get update
RUN python3 -m pip install --upgrade pip
RUN pip install llvmlite --ignore-installed && \
    pip install librosa==0.8.0 
RUN pip install pydub && \
    pip install faiss-gpu

# RUN apt-get install libsox-fmt-all libsox-dev sox > /dev/null
RUN python -m pip install torchaudio > /dev/null && \
    python -m pip install git+https://github.com/facebookresearch/WavAugment.git > /dev/null

RUN pip install opencv-python && \
    pip install visdom

COPY . /model
WORKDIR /model

CMD /bin/bash