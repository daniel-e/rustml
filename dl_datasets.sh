#!/bin/bash

set -e

RUSTMLPATH=~/.rustml

echo
if [ ! -e $RUSTMLPATH ]; then
	mkdir $RUSTMLPATH
fi

# mnist dataset of handwritten digits
echo -e "\e[1;34mdownloading MNIST dataset of handwritten digits ...\e[0m"
if [ ! -e $RUSTMLPATH/datasets/mnist_digits ]; then
	mkdir -p $RUSTMLPATH/datasets/mnist_digits
fi
for i in  \
	"t10k-images-idx3-ubyte.gz" \
	"t10k-labels-idx1-ubyte.gz" \
	"train-images-idx3-ubyte.gz" \
	"train-labels-idx1-ubyte.gz" \
; do
	URL="https://github.com/daniel-e/rustml/blob/master/datasets/mnist_digits/$i?raw=true"
	echo "> downloading $i ..."
	wget -q -O $RUSTMLPATH/datasets/mnist_digits/$i $URL
done

# --------------------------------------------

echo -e "\e[1;34mdownloading example videos ...\e[0m"
if [ ! -e $RUSTMLPATH/datasets/videos ]; then
	mkdir -p $RUSTMLPATH/datasets/videos
fi
for i in  \
	"day_and_night_in_gray.avi" \
	"day_and_night_in_gray_mask_sky.png" \
; do
	URL="https://github.com/daniel-e/rustml/blob/master/datasets/videos/$i?raw=true"
	echo "> downloading $i ..."
	wget -q -O $RUSTMLPATH/datasets/videos/$i $URL
done
echo -e "\e[1;32mdone\e[0m"	


