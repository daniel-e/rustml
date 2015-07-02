extern crate time;

pub mod matrix;
pub mod blas;

use matrix::*;

fn main() {
    type T = f32;

    // Create random matrices.
    println!("Creating matrices ...");
    let a = Matrix::<T>::fill(1.0, 6000, 6000);
    let b = Matrix::<T>::fill(2.0, 6000, 6000);

    println!("Computing ...");
    let t1 = time::now();
    let c = a * b;
    let t2 = time::now();
    println!("{}", t2 - t1);
}

