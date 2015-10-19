1;
more off;

% create a dataset where the points cannot be separated by a linear
% classifier
[x, y] = l4_dataset(100);

% show the dataset
clf;
plot_dataset(x, y);

%%
display("press enter to continue...");
pause;
%%

display("training a neural network...");

[yv, mapping] = convert(y);
[p1, p2, err] = nn_train(2, 5, size(mapping, 1), x, yv);

%%
display("press enter to plot the decision boundaries...");
pause;
%%

clf;
plot(err, 'linewidth', 2);
grid on;
title("Learning curve");

%%
display("press enter to continue...");
pause;
%%

%% contour plot
tx = ty = linspace(-1, 2, 40);
[xx, yy] = meshgrid(tx, ty);
zz = zeros(size(xx));
for r = 1:size(xx, 1)
	for c = 1:size(xx, 2)
		[mxval, mxidx] = max(nn_predict(p1, p2, [xx(r, c); yy(r, c)]));
		zz(r, c) = mxidx;
	end
end
plot_dataset(x, y);
contour(tx, ty, zz, 'linewidth', 2);
