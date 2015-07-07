# Rustml

Library for doing machine learning in Rust. 

The documentation for this project can be found  
[here](http://daniel-e.github.io/rustml/rustml).

## Features 

* powerful matrix and vector support with BLAS bindings for high performance 
* classification with k-nearest neighbord
* access to MNIST database of handwritten digits
* parse and create CSV files
* statistical functions like mean and variance for vectors and matrices
* reading gzip compressed files

## Example: matrix multplication

```rust
use rustml::*;

fn main() {

    let a = mat![
        1.0, 2.0; 
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

### Other examples

You can find orher examples in the directory `examples`. These examples can be executed with
Cargo as follows:

```bash
cargo run --example vector_addition
cargo run --example mnist_digits
cargo run --example matrix_multiplication
``` 
