#!/bin/bash

echo "clean"
cargo clean

echo "building lib ..."
cargo build

for i in $(find examples/ -name "*.rs" | cut -d/ -f2 | cut -d. -f1);
do
	echo "example $i ..."
	cargo run --example $i
done

echo "doc"
cargo doc

echo "test"
cargo test

