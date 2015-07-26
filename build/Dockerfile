# sudo docker build -t rustml/build .

FROM ubuntu:14.04
RUN apt-get update
RUN apt-get -y install build-essential git aptitude wget unzip libblas-dev
ADD runner.sh /root/
