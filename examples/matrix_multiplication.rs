extern crate time;
#[macro_use]
extern crate rustml;

use rustml::Matrix;

fn measure_time() {
    type T = f32;

    // Create random matrices.
    println!("Creating matrices ...");
    let a = Matrix::<T>::fill(1.0, 6000, 6000);
    let b = Matrix::<T>::fill(2.0, 6000, 6000);

    println!("Computing ...");
    let t1 = time::now();
    let _c = a * b;
    let t2 = time::now();
    println!("Time to compute a*b = {}", t2 - t1);
}

fn small_matrix_multiplication() {
    let a = mat![1.0, 2.0; 3.0, 4.0];
    let b = mat![5.0, 6.0; 7.0, 8.0];
    let c = a * b;
    println!("{}", c);
}

fn main() {
    small_matrix_multiplication();
    measure_time();
}

