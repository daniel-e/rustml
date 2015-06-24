pub mod matrix;
//pub mod lof;
pub mod distance;
pub mod norm;
pub mod blas;

use matrix::*;

fn main() {

    let a = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];
    let b = mat![5.0, 7.0; 6.0, 2.0];
    let c = a * b;

    println!("{}", c);
}


