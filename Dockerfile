FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV PYTHON_VERSION 3.7.4
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        make build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev python-openssl git wget curl

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && $PYENV_ROOT/plugins/python-build/install.sh \
    && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
    && rm -rf $PYENV_ROOT

ADD . /src
WORKDIR /src
RUN pip install pipenv && pipenv install

