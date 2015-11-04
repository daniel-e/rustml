1;
addpath("/home/dz/ml/libsvm-3.20/matlab/");
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

display("training svm...");

model = svmtrain(y, x, '');
[labels, acc, p] = svmpredict(y, x, model, '');

%% contour plot
tx = ty = linspace(-1, 2, 40);
[xx, yy] = meshgrid(tx, ty);
zz = zeros(size(xx));
for r = 1:size(xx, 1)
	for c = 1:size(xx, 2)
		[l, a, p] = svmpredict([0], [xx(r, c), yy(r, c)], model, '');
		zz(r, c) = l;
	end
end
plot_dataset(x, y);
contour(tx, ty, zz, 'linewidth', 2);
