function [y, mapping] = convert(labels)
	l = labels(:);    % labels as column vector
	u = unique(l);    % column vector
	n = size(u, 1);   % number of unique labels
	m = size(l, 1);   % number of labels

	% create mapping
	mapping = zeros(n, n + 1);
	for i = 1:n
		mapping(i, 1) = u(i, 1); % first element is the label
		mapping(i, 1 + i) = 1;
	end

	y = zeros(m, n);
	for i = 1:m
		rows = mapping(:, 1) == l(i, 1);
		y(i, :) = mapping(rows, 2:end);
	end
end
