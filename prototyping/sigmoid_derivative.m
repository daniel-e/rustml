function [y] = sigmoid_derivative(x)
	y = sigmoid(x) .* (1 - sigmoid(x));
end
