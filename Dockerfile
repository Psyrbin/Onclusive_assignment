FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y python3 python3-pip

WORKDIR /workdir
COPY requirements.txt .

RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt

RUN python3 -c "import nltk; nltk.download('punkt')"
