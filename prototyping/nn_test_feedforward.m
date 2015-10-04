1;

params1 = [0.1, 0.2, 0.4; 0.2, 0.1, 2.0];
params2 = [0.8, 1.2, 0.6; 0.4, 0.5, 0.8; 1.4, 1.5, 2.0];
x = [0.5, 1.2, 1.5]';

z0 = x;
a0 = x;
z1 = params1 * x
a1 = [1; sigmoid(z1)]
z2 = params2 * a1
a2 = sigmoid(z2)

t = [2.7, 3.1, 1.5]';   % target output
d0 = (a2 - t) .* sigmoid_derivative(z2)
d1 = (params2' * d0)(2:end, :) .* sigmoid_derivative(z1)
