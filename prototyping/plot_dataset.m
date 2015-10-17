function plot_dataset(x)
	% TODO more colors
	colors = [
		0, 1, 1;
		1, 0, 0;
		0, 0, 1;
		1, 1, 0
	];
	clf;
	for i = 0:3
		s = x(:, 1) == i;
		plot(x(s, 2), x(s, 3), 'o', 'markersize', 8, 'markerfacecolor', colors(i + 1, :), 'markeredgecolor', 'k', 'linewidth', 2);
		hold on;
	end
	grid on;
	axis([-1, 2, -1, 2]);
end
