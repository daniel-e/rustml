#!/bin/bash

echo "."
cargo clean

cd examples
for i in *; do
	echo $i
	cd $i
	cargo clean
	cd ..
done
cd ..

echo "."
cargo build

cd examples
for i in *; do
	echo $i
	cd $i
	cargo build
	cd ..
done
cd ..

echo "doc"
cargo doc
echo "test"
cargo test

