#!/bin/bash

set -e

function installrust {
	rm -rf /opt/rust

	pushd /tmp >/dev/null
	wget -q https://static.rust-lang.org/dist/$1.tar.gz
	tar xzf $1.tar.gz
	cd $1 >/dev/null
	./install.sh --prefix=/opt/rust
	popd >/dev/null
}

function testrustml {
	export PATH=/opt/rust/bin/:$PATH
	export LD_LIBRARY_PATH=/opt/rust/lib/:$LD_LIBRARY_PATH

	rm -rf /tmp/rustml-master/ /tmp/rustml/ /root/.rustml/

	# ------ master
	cd /tmp
	wget -q https://github.com/daniel-e/rustml/archive/master.zip
	unzip master.zip
	cd rustml-master/

	cargo build
	cargo test --lib

	./dl_datasets.sh
	./build_all.sh
	cd ..
	rm -rf rustml-master

	# ------- dev branch
	git clone https://github.com/daniel-e/rustml.git
	cd rustml
	git checkout dev.0.0.3
	./build_all.sh
}

installrust rust-1.1.0-x86_64-unknown-linux-gnu
testrustml
installrust rust-beta-x86_64-unknown-linux-gnu
testrustml

echo "done"
