#!/bin/bash

for i in $(find examples/ -name "*.rs" | cut -d/ -f2 | cut -d. -f1);
do
	echo "example $i ..."
	cargo run --example $i
done

