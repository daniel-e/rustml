The source code of atlas and lapack has been downloaded from:
http://math-atlas.sourceforge.net/
http://www.netlib.org/lapack/

Install
-------

for i in {0..7}; do cpufreq-set -g performance -c $1; done
# /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

tar xjf atlas3.10.2.tar.bz2
mkdir ATLAS/build
cd ATLAS/build
../configure --nof77 --shared -b 64 -D c -DPentiumCPS=3400 --prefix=/opt/atlas --with-netlib-lapack-tarfile=../../lapack-3.5.0.tgz
make build
mkdir -p /opt/atlas/lib/
cp lib/lib*.so /opt/atlas/lib/
cd ../..
rm -rf ATLAS

---

LD_PRELOAD=/usr/lib/libblas.so.3 cargo run --bin matrix_mul
PT19.900532926S

LD_PRELOAD=/usr/lib/libatlas.so.3 cargo run --bin matrix_mul
PT19.889701623S

LD_PRELOAD=/opt/atlas/lib/libsatlas.so cargo run --bin matrix_mul
PT13.021963039S

LD_PRELOAD=/opt/atlas/lib/libtatlas.so cargo run --bin matrix_mul
PT6.459068706S

