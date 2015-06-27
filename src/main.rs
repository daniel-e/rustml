pub mod io;
pub mod matrix;
//pub mod lof;
pub mod distance;
pub mod norm;
pub mod blas;
pub mod knn;
pub mod csv;

use matrix::*;

fn main() {

    let m1 = Matrix::<f64>::random::<f64>(5, 2);
    let m2 = Matrix::<f64>::random::<f64>(5, 2);
}


