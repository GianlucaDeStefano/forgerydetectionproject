# syntax=docker/dockerfile:1
# https://docs.docker.com/develop/develop-images/multistage-build/

ARG DROPBEAR_DEB=dropbear_2022.82-0.1_amd64.deb

FROM projects.cispa.saarland:5005/css/ngc/pytorch:22.07-py3 AS builder
ARG DROPBEAR_DEB
RUN sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get build-dep -y dropbear
RUN git clone https://github.com/mkj/dropbear.git && cd dropbear && git checkout DROPBEAR_2022.82 && dpkg-buildpackage -uc -us

FROM projects.cispa.saarland:5005/css/ngc/pytorch:22.07-py3
ARG DROPBEAR_DEB
COPY --from=builder /workspace/$DROPBEAR_DEB ./
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y libtomcrypt1 libtommath1
RUN ls -al $DROPBEAR_DEB && dpkg -i $DROPBEAR_DEB && rm $DROPBEAR_DEB
ENV DEBIAN_FRONTEND=noninteractive

RUN apt install nvidia-cuda-toolkit -yq

COPY requirements.txt /opt/app/requirements.txt

WORKDIR /opt/app

RUN pip install -r requirements.txt

RUN apt-get install -yq ffmpeg
RUN apt-get install -yq libsm6
RUN apt-get install -yq libxext6

RUN apt-get install htop
