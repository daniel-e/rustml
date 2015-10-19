% plot_dataset(x, y)
%
% Creates a two-dimensional plot of the points in x with the labels in y.
% 
%
% RETURNS
% 
%  x   A matrix which contains in each row the coordinates of one point.
%  y   A column vector which contains the labels for the points.
function plot_dataset(x, y)

	colors = [
		1, 1, 0;
		1, 0, 0;
		0, 0, 1;
		1, 0, 1;
		0, 1, 1;
		0, 1, 0;
		0, 0, 0
	];

	u = unique(y)(:);  % unique labels
	assert(size(u, 1) <= size(colors, 1), 'Number of unique labels is larger than number of colors.');

	for i = 1:size(u, 1)
		s = y(:, 1) == u(i, 1);
		plot(x(s, 1), x(s, 2), 'o', 'markersize', 8, 'markerfacecolor', colors(i, :), 'markeredgecolor', 'k', 'linewidth', 2);
		hold on;
	end
	grid on;
end
