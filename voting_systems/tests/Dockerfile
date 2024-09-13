FROM python:3.7-buster

ENV DEBIAN_FRONTEND noninteractive

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install numpy pytest

COPY voting_systems/tests/requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

COPY . /voting_systems

WORKDIR /voting_systems

ENTRYPOINT ["/usr/local/bin/pytest", "-s"]
