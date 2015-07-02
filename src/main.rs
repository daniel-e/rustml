extern crate rand;
extern crate time;

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
use vectors::group;
use datasets::MnistDigits;

fn main() {
    type T = f32;

    let k = 5;

    println!("Reading training data ...");
    let (training, training_labels) = MnistDigits::training_set::<T>().unwrap();

    println!("Reading test data ...");
    let (test, test_labels) = MnistDigits::test_set::<T>().unwrap();

    println!("Classifying ...");
    let r = test.row_iter().zip(test_labels.iter()).take(10)
        .map(|(row, label)| {
            let idx = knn_scan(&training, row, k, |x, y| Euclid::compute(x, y).unwrap()).unwrap();

            let mut targets: Vec<u8> = idx.iter().map(|pos| training_labels.get(*pos).unwrap()).cloned().collect();
            targets.sort_by(|a, b| a.cmp(&b));
            let mut r = group(&targets);
            r.sort_by(|a, b| a.1.cmp(&b.1));

            (label, r.last().unwrap().0)
        });

    let t1 = time::now();
    for (x, y) in r {
        println!("{} {}", x, y);
    }
    let t2 = time::now();
    println!("{}", t2 - t1);
}


