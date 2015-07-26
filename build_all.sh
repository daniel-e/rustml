#!/bin/bash

echo "clean"
cargo clean

echo "building lib ..."
cargo build

echo "building examples ..."
./build_examples.sh

echo "doc"
cargo doc

echo "test"
cargo test
