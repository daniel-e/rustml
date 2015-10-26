1;

more off;
printf('reading digits ...\n')
load('data/mnist.txt.zip');

m = zeros(2 * 28, 5 * 28);
for i = 0:9
	r = trainY == i;
	v = sum(trainX(r, :)) / sum(r);
	y = floor(i / 5);
	x = mod(i, 5);
	m(y * 28 + 1:y * 28 + 1 + 27, x * 28 + 1:x * 28 + 1 + 27) = reshape(v, 28, 28)';
end

imagesc(m);
