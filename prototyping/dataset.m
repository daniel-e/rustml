function [x, y] = dataset()
	rand('seed', 1);
	n = 100;
	s = 0.24;
	% around (0, 0)
	a = [normrnd(0, s, n, 2)];
	% around (1, 1)
	b = [normrnd(0, s, n, 2) + 1];
	% around (1, 0)
	c = [normrnd(0, s, n, 1) + 1, normrnd(0, s, n, 1)];
	% around (0, 1)
	d = [normrnd(0, s, n, 1), normrnd(0, s, n, 1) + 1];
	x = [a; b; c; d];
	y = [zeros(n, 1); ones(n, 1); ones(n, 1) + 1; ones(n, 1) + 2];
end
