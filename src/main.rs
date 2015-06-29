extern crate rand;

pub mod io;
pub mod matrix;
//pub mod lof;
pub mod distance;
pub mod norm;
pub mod blas;
pub mod knn;
pub mod csv;

use matrix::*;
use rand::{weak_rng, Rng};

fn knn_classify<It: Iterator>(it: It, v: &[f64]) -> f64 {
    1.0
}

fn main() {

    let m = Matrix::<f64>::random::<f64>(100, 2);

    let mut v: Vec<usize> = (0..m.rows()).collect();
    weak_rng().shuffle(&mut v);
    let (training, test) = v.split_at(50);
    let r = m.row_iter_of(test)
        .map(|r| {
            let (label, row) = r.split_at(1);
            (label,
             knn_classify(m.row_iter_of(training), row)
            )
        });
}


