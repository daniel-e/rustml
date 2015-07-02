## Creating a build environment to run rustml

### Run a clean Ubuntu 14.04 in a docker container

```bash
sudo ./docker run -t -i ubuntu:14.04 /bin/bash
```

### Install the required dependencies

```bash
apt-get update
apt-get -y install build-essential git screen aptitude wget unzip libblas-dev 
```

### Install rust

```bash
wget https://static.rust-lang.org/dist/rust-1.1.0-x86_64-unknown-linux-gnu.tar.gz
tar xzf rust-1.1.0-x86_64-unknown-linux-gnu.tar.gz
cd rust-1.1.0-x86_64-unknown-linux-gnu
./install.sh --prefix=/opt/rust
cd ..
echo "export PATH=/opt/rust/bin/:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/opt/rust/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc
. ~/.bashrc
```

### Download rustml and run the example for doing a matrix multiplication

```bash
wget https://github.com/daniel-e/rustml/archive/master.zip
unzip master.zip
cd rustml-master/
cargo run --bin matrix_mul

# measured time
# PT116.067883051S
```

## Improve performance

You can improve the performance by a factor of 6 by simply installing libatlas.

```bash
apt-get -y install libatlas3-base
cargo run --bin matrix_mul
PT19.576345377S
```

## Improve performance even further

You can improve the performance even further by a factor of >2 if you compile ATLAS by your own optimized for your architecture.

```bash
cd atlas
tar xjf atlas3.10.2.tar.bz2
mkdir ATLAS/build
cd ATLAS/build
../configure --nof77 --shared -b 64 -D c -DPentiumCPS=3400 --prefix=/opt/atlas --with-netlib-lapack-tarfile=../../lapack-3.5.0.tgz
make build
mkdir -p /opt/atlas/lib
cp lib/*.so /opt/atlas/lib/
cd ..
```

```bash
# single threaded version of ATLAS
LD_PRELOAD=/opt/atlas/lib/libsatlas.so cargo run --bin matrix_mul
PT12.945538954S
```

```bash
# multi threaded version of ATLAS
LD_PRELOAD=/opt/atlas/lib/libtatlas.so cargo run --bin matrix_mul
PT6.475673429S
```
