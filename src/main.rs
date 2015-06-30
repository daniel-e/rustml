extern crate rand;

pub mod io;
pub mod matrix;
//pub mod lof;
pub mod distance;
pub mod norm;
pub mod blas;
pub mod knn;
pub mod csv;
pub mod datasets;

use matrix::*;

fn knn_classify(m: &Matrix<f64>, example: &[f64], k: usize) -> f64 {

    for row in m.row_iter() {
        let (label, r) = row.split_at(1);
        
    }
    1.0
}

fn main() {

    let k = 3;

    println!("Reading training data ...");
    let training = datasets::MnistDigits::training_set().unwrap();

    println!("Reading test data ...");
    let test = datasets::MnistDigits::test_set().unwrap();

    println!("Classifying ...");
    let r = test.row_iter()
        .map(|r| {
            let (label, row) = r.split_at(1);
            (label,
             knn_classify(&training, row, k)
            )
        });

}


