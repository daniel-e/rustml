For a documentation of the interfaces have a look at the 
[rustml documentation](http://daniel-e.github.io/rustml/rustml).

# Example: matrix multplication

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

# Examples

You can find orher examples in the directory `examples`. These examples can be executed with
Cargo as follows:

```bash
cargo run --example vector_addition
cargo run --example mnist_digits
cargo run --example matrix_multiplication
``` 
