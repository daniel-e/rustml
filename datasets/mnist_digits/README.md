Dataset of digits from http://yann.lecun.com/exdb/mnist/

reduced dataset

# remove 10000 examples
dd if=train-images-idx3-ubyte of=train-images-idx3-ubyte.50k bs=16 count=2450001
gzip -9 train-images-idx3-ubyte.50k
dd if=train-labels-idx1-ubyte of=train-labels-idx1-ubyte.50k bs=1 count=50008
gzip -9 train-labels-idx1-ubyte.50k

