The following packages needs to be downloaded:
* http://downloads.sourceforge.net/project/math-atlas/Stable/3.10.2/atlas3.10.2.tar.bz2
* http://www.netlib.org/lapack/lapack-3.5.0.tgz

Install
-------

```
# turn off CPU throttling
# see http://math-atlas.sourceforge.net/atlas_install/node5.html
for i in {0..7}; do cpufreq-set -g performance -c $i; done

# install ATLAS
tar xjf atlas3.10.2.tar.bz2
mkdir ATLAS/build
cd ATLAS/build
../configure --nof77 --shared -b 64 -D c -DPentiumCPS=3400 --prefix=/opt/atlas --with-netlib-lapack-tarfile=../../lapack-3.5.0.tgz
make build
mkdir -p /opt/atlas/lib/
cp lib/lib*.so /opt/atlas/lib/
cd ../..
rm -rf ATLAS
```

---

LD_PRELOAD=/usr/lib/libblas.so.3 cargo run --bin matrix_mul
PT19.900532926S

LD_PRELOAD=/usr/lib/libatlas.so.3 cargo run --bin matrix_mul
PT19.889701623S

LD_PRELOAD=/opt/atlas/lib/libsatlas.so cargo run --bin matrix_mul
PT13.021963039S

LD_PRELOAD=/opt/atlas/lib/libtatlas.so cargo run --bin matrix_mul
PT6.459068706S

