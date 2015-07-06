#!/bin/bash

echo "clean"
cargo clean

echo "building lib ..."
cargo build

for i in vector_addition mnist_digits matrix_multiplication; do
	echo "example $i ..."
	cargo run --example $i
done

echo "doc"
cargo doc

echo "test"
cargo test

