#!/bin/bash

set -e

function gen {
	cargo run --example linear_regression
	convert -trim /tmp/linreg_plot.png doc_data/linreg_plot.png
	cargo run --example image_grid
	convert -resize x246 /tmp/grid.png doc_data/digits_grid.png
	cargo run --example gradient_descent
	convert -trim /tmp/3dplot.png doc_data/gradient_descent.png
}

echo "Generating doc ..."
cargo doc

if [ "$1" == "all" ]; then
	gen
fi

cp doc_data/* target/doc/
