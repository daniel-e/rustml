cargo run --example mnistdigits2matrix
cd /tmp
start octave

trainX = load('mnist_training.txt');
trainY = load('mnist_training_labels.txt');
testX = load('mnist_test.txt');
testY = load('mnist_test_labels.txt');

save -z "mnist.txt.zip" trainX trainY testX testY
