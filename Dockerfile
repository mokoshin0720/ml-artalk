FROM python:3.9
# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN mkdir /artalk
COPY . /artalk/
WORKDIR /artalk/

# RUN apt update
# RUN apt install -y wget

# RUN apt-key del 7fa2af80
# RUN apt-get install -y wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb 
RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 
RUN apt-get install -y libgl1-mesa-dev wget unzip libglib2.0-0
RUN apt install -y cmake

RUN pip install --upgrade pip
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
# RUN wget https://fonts.google.com/download?family=Noto%20Sans%20JP -O /tmp/fonts_noto.zip && \
#     mkdir -p /usr/share/fonts &&\
#     unzip /tmp/fonts_noto.zip -d /usr/share/fonts
# RUN python3 -m spacy download en_core_web_lg