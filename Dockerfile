FROM python:3

RUN mkdir /build
COPY . /build/
WORKDIR /build/

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm