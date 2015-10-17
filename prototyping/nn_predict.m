% TODO doc
function [label] = nn_predict(p1, p2, x)
	z2 = p1 * x;
	a2 = [1; sigmoid(z2)];
	z3 = p2 * a2;
	a3 = sigmoid(z3);
	label = a3;
end
