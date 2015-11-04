1;

more off;

if exist('trainX') == 0
	printf('reading digits ...\n')
	load('data/mnist.txt.zip');
end

avg = [];
m = zeros(2 * 28, 5 * 28);
for i = 0:9
	r = trainY == i;
	v = sum(trainX(r, :)) / sum(r);
	avg = [avg; v];
	y = floor(i / 5);
	x = mod(i, 5);
	m(y * 28 + 1:y * 28 + 1 + 27, x * 28 + 1:x * 28 + 1 + 27) = reshape(v, 28, 28)';
end

imagesc(m);


m = size(testX, 1);   % number of test examples

c = 0;
for i = 1:m
	e = [];
	for j = 1:10
		e = [e; sumsq(testX(i, :) - avg(j, :))];
	end
	[mnval, mnidx] = min(e);
	if mnidx - 1 == testY(i, 1)
		c += 1;
	end
end
c
c / m
