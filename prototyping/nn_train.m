% TODO doc, iter, alpha
function [p1, p2, err] = nn_train(x, y, settings)
	iter  = settings.iter;
	alpha = settings.alpha;
	i = settings.input;
	h = settings.hidden;
	o = settings.output;
	if isfield(settings, 'progress')
		w = waitbar(0);
	end

	% number of examples
	m = size(x, 1);
	% parameters from input layer to hidden layer
	p1 = rand(h, i) - 0.5;
	% parameters from hidden layer + bias unit to output layer
	p2 = rand(o, h + 1) - 0.5;
	% learning curve
	err = [];

	for j = 1:iter
		if exists('w') == 1
			waitbar(j / iter, w);
		end

		% feedforward
		z2 = x * p1';
		a2 = [ones(m, 1), sigmoid(z2)];
		z3 = a2 * p2';
		a3 = sigmoid(z3);

		% row i, col j in a3 contains the output of neuron j in the
		% output layer for example i

		err = [err; sum(sumsq(y - a3)) / (2 * m)];

		% backprop
		dd1 = zeros(size(p1));
		dd2 = zeros(size(p2));
		for i = 1:m
			d3 = (a3(i, :)' - y(i, :)') .* sigmoid_derivative(z3(i, :)');
			d2 = (p2' * d3)(2:end, :) .* sigmoid_derivative(z2(i, :)');
			dd2 = dd2 + d3 * a2(i, :);
			dd1 = dd1 + d2 * x(i, :);
		end
		p1 = p1 - alpha * dd1 / m;
		p2 = p2 - alpha * dd2 / m;
	end

	if exists('w') == 1
		close(w);
	end
end

