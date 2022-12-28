FROM python:3.9

RUN mkdir /artalk
COPY . /artalk/
WORKDIR /artalk/

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