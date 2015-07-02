This is an introduction how to use rustml from a scratch Ubuntu installation. 


sudo ./docker run -t -i ubuntu:14.04 /bin/bash

```bash
apt-get update
apt-get install build-essential git screen aptitude wget unzip libblas-dev 

wget https://static.rust-lang.org/dist/rust-1.1.0-x86_64-unknown-linux-gnu.tar.gz
tar xzf rust-1.1.0-x86_64-unknown-linux-gnu.tar.gz
cd rust-1.1.0-x86_64-unknown-linux-gnu
./install.sh --prefix=/opt/rust
cd ..
echo "export PATH=/opt/rust/bin/:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/opt/rust/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc
. ~/.bashrc

wget https://github.com/daniel-e/rustml/archive/master.zip
unzip master.zip
cd rustml-master/
cargo run --bin matrix_mul
PT116.067883051S
```

# Improve performance

```
apt-get install libatlas3-base
cargo run --bin matrix_mul
PT19.576345377S
```

# Improve performance even more

