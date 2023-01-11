FROM nvcr.io/nvidia/tensorflow:21.04-tf2-py3

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN pip install -r requirements.txt