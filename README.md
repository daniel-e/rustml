# Rustml

Rustml is a library for doing machine learning in Rust. 

The documentation of the project with a descprition of the modules can be found [here](http://daniel-e.github.io/rustml/rustml).

## Features 

* powerful matrix and vector operations with BLAS bindings for high performance computing
* k-nearest neighbord classification algorithm
* neural networks
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

You can find other examples in the directory `examples`. Each example can be executed with `cargo run --example <example name>` where `<example name>` is one of the following:

* `gradient_descent`: uses gradient descent to find the minimum of a function
* `image_grid`: how to arrange a set of images into a grid
* `linear_regression`: use linear regression to approximate the points of an unknown function
* `matrix_multiplication`: guess what ;)
* `mnist_digits`: shows how to load the MNIST database of handwritten digits
* `neuralnetwork`: trains a neural network to compute the XOR function and plots the decision boundaries
* `octave_plot`: shows how to plot things with Octave
* `plots`: creates some plots for the online documentation
* `scale_matrix`: demonstrates how to scale features
* `vector_addition`: demonstrates how to add vectors
* `video_histogram`: how to select specific regions of a video and compute the histogram

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

