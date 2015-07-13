# Rustml

Rustml is a library for doing machine learning in Rust. 

The documentation of the project can be found [here](http://daniel-e.github.io/rustml/rustml).

## Features 

* powerful matrix and vector support with BLAS bindings for high performance 
* classification with k-nearest neighbord
* easy access to MNIST database of handwritten digits
* parse and create CSV files
* statistical functions like mean and variance for vectors and matrices
* reading gzip compressed files
* distance metrics

## Using rustml - example matrix multplication

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

You can find orher examples in the directory `examples`. These examples can be executed with
Cargo as follows:

```bash
cargo run --example vector_addition
cargo run --example mnist_digits
cargo run --example matrix_multiplication
``` 

## Datasets

Rustml comes with the MNIST database of handwritten digits and provides
an API to access this database. In prior to use the dataset you have to
install it into you home path. This can be easily done as follows:

```
# download the install script
wget -q https://github.com/daniel-e/rustml/blob/master/dl_datasets.sh
chmod +x dl_datasets.sh

# download the datasets and install them into ~/.rustml/
./dl_datasets.sh
```

