FROM ros:iron

ARG CUDNN_ARCH="linux-x86_64"

# install cuda keyring
RUN apt update && \
    apt install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb

# install apt dependencies
RUN apt update && \
    # libportaudio2 (for python-sounddevice)
    apt install -y libportaudio2 && \
    # cuda (also ship the required libcublas)
    apt install -y cuda-toolkit && \
    # python3
    apt install -y python3 python3-pip python-is-python3

# install cudnn8 from tarball
RUN wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/${CUDNN_ARCH}/cudnn-${CUDNN_ARCH}-8.9.5.30_cuda12-archive.tar.xz && \
    tar -xvf cudnn-${CUDNN_ARCH}-8.9.5.30_cuda12-archive.tar.xz && \
    mv cudnn-${CUDNN_ARCH}-8.9.5.30_cuda12-archive/include/* /usr/local/cuda/include/ && \
    mv cudnn-${CUDNN_ARCH}-8.9.5.30_cuda12-archive/lib/* /usr/local/cuda/lib64/

# install python dependencies
RUN pip install faster-whisper sounddevice

WORKDIR /asr

# download large-v3 model
RUN mkdir model
RUN wget https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json -P model/
RUN wget https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json -P model/
RUN wget https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json -P model/
RUN wget https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json -P model/
RUN wget https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin -P model/

# copy online asr script
COPY whisper_online_mic.py .

ENV PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=/opt/ros/iron/lib/x86_64-linux-gnu:/opt/ros/iron/lib:$LD_LIBRARY_PATH

ENTRYPOINT [ "python", "whisper_online_mic.py", "--model_dir", "model/" ]