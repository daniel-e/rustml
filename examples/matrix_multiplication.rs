extern crate time;
extern crate rustml;

use rustml::Matrix;

fn main() {
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

