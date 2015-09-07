# Rustml

Rustml is a library for doing machine learning in Rust. 

The documentation of the project with a descprition of the modules can be found [here](http://daniel-e.github.io/rustml/rustml).

## Features 

* powerful matrix and vector support with BLAS bindings to provide very high performance 
* k-nearest neighbord classification algorithm
* DBSCAN clustering algorithm
* gradient descent for minimizing functions
* linear regression
* easy access to MNIST database of handwritten digits via an the rustml dataset package
* parse and create CSV files
* statistical functions like mean and variance for vectors and matrices
* reading gzip compressed files
* distance metrics
* OpenCV binding to read images and videos plus interfaces for simplify feature extraction from images and videos (e.g. select pixels from images or frames of a video via a mask)
* scaling of feature vectors and matrices
* multidimensional sliding windows
* examples

## Using rustml from scratch - example matrix multplication

Create a new project with cargo:

```bash
cargo new example --bin
```

A new directory `example` is created. Change into this directory and add the following lines `Cargo.toml`:
```
[dependencies.rustml]
git = "https://github.com/daniel-e/rustml/"
```

or the following dependency:
```
[dependencies]
rustml = "*"
```

Edit the file `main.rs` in the `src` directory.

```rust
#[macro_use] extern crate rustml;

use rustml::*;

fn main() {
    let a = mat![
        1.0f32, 2.0;
        3.0, 4.0;
        5.0, 6.0
    ];
    let b = mat![
        5.0, 7.0;
        6.0, 2.0
    ];
    let c = (a * b).unwrap();

    println!("{}", c);
}
```

Now, in the `example` directory run the example with `cargo run`.

### Other examples

You can find other examples in the directory `examples`. These examples can be executed with
Cargo as follows:

```bash
cargo run --example vector_addition
cargo run --example matrix_multiplication
cargo run --example scale_matrix

# the following examples require the rustml dataset package (see below)
cargo run --example video_histogram
cargo run --example mnist_digits
``` 

## Rustml datasets package

The rustml dataset package needs to be installed separately. The package currently contains
the MNIST database of handwritten digits and videos for the examples. Download the following
script which will download and install the package in your home in the directory
`~/.rustml/`.

```bash
# download the install script
wget -q https://raw.githubusercontent.com/daniel-e/rustml/master/dl_datasets.sh
chmod +x dl_datasets.sh

# download the datasets and install them into ~/.rustml/
./dl_datasets.sh
```

