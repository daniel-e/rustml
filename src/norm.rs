extern crate libc;

use ::blas::cblas_dnrm2;

pub trait Norm<T> {
    fn compute(a: &[T]) -> T;
}

pub struct L2Norm;

impl Norm<f64> for L2Norm {

    fn compute(a: &[f64]) -> f64 {
        unsafe {
            cblas_dnrm2(
                a.len()    as libc::c_int,
                a.as_ptr() as *const libc::c_double,
                1
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Norm, L2Norm};

    #[test]
    fn test_l2nrom() {

        let a = vec![1.0, 2.0, 3.0];
        assert!(L2Norm::compute(&a) - 3.741657 <= 0.000001);
    }
}

