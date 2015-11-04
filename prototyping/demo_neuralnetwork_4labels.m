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
s = struct(
	"iter", 500, "alpha", 5.0, 
	"input", 2, "hidden", 5, "output", size(mapping, 1),
	"progress", 1);
[p1, p2, err] = nn_train(x, yv, s);

clf;
plot(err, 'linewidth', 2);
grid on;
title("Learning curve");

%%
display("press enter to continue to plot decision boundaries ...");
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
