extern crate rand;

pub mod io;
pub mod matrix;
pub mod distance;
pub mod norm;
pub mod blas;
pub mod knn;
pub mod csv;
pub mod datasets;
pub mod vectors;

use distance::*;
use knn::knn_scan;

fn main() {

    let k = 5;

    println!("Reading training data ...");
    let training = datasets::MnistDigits::training_set().unwrap();

    println!("Reading test data ...");
    let test = datasets::MnistDigits::test_set().unwrap();

    println!("Classifying ...");
    let r = test.row_iter().take(10)
        .map(|r| {
            let (l, row) = r.split_at(1);
            let label = l.get(0).unwrap().clone();
            (label,
             knn_scan(&training, row, k, |x, y| Euclid::compute(x, y).unwrap()).unwrap()
            )
        });

    for (x, y) in r {
        println!("{} {}", x, y);
    }
}


