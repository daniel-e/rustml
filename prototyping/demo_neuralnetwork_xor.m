1;

% TODO rename: demo_xor_neuralnetworks

[x, y] = dataset();
plot_dataset(x, y);

display("press enter to continue...");
pause;

[p1, p2, err] = nn_train(2, 10, 1, x, y);

clf;
plot(err);

display("press enter to continue...");
pause;

tx = ty = linspace(-1, 2, 40);
[xx, yy] = meshgrid(tx, ty);
zz = zeros(size(xx));
for r = 1:size(xx, 1)
	for c = 1:size(xx, 2)
		zz(r, c) = nn_predict(p1, p2, [xx(r, c); yy(r, c)]);
	end
end
plot_dataset(x, y);
contour(tx, ty, zz, 'linewidth', 2);
