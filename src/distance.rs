extern crate libc;

use ::matrix::*;
use ::norm::{L2Norm, Norm};
use ::blas::cblas_daxpy;

pub trait Distance<T> {
    fn compute(a: &[T], b: &[T]) -> Option<T>;
}

pub struct Euclid;

impl Distance<f64> for Euclid {

    fn compute(a: &[f64], b: &[f64]) -> Option<f64> {

        if a.len() != b.len() {
            return None;
        }

        // c = b.clone() does not work here because cblas_daxpy
        // modifies the content of c and cloned() on a slice does
        // not create a copy.
        let c: Vec<f64> = b.iter().cloned().collect();

        unsafe {
            cblas_daxpy(
                a.len()     as libc::c_int,
                -1.0        as libc::c_double,
                a.as_ptr()  as *const libc::c_double,
                1           as libc::c_int,
                c.as_ptr()  as *mut libc::c_double,
                1           as libc::c_int
            );
        }
        Some(L2Norm::compute(&c))
    }
}

pub fn all_pair_distances<'t>(m: &Matrix<f64>) -> Matrix<f64> {

    let mut r = Matrix::fill(0.0, m.rows(), m.rows());

    for i in 0..m.rows() {
        for j in (i + 1)..m.rows() {
            let d = Euclid::compute(&m.row(i).unwrap(), &m.row(j).unwrap()).unwrap();
            r.set(i, j, d);
            r.set(j, i, d);
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::{Distance, Euclid, all_pair_distances};
    use ::matrix::*;

    #[test]
    fn test_euclid() {

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 5.0, 13.0];
        let c = vec![2.0, 5.0, 13.0];
        let d = vec![1.0, 2.0, 3.0];
        assert!(Euclid::compute(&a, &b).unwrap() - 10.488088 <= 0.000001);
        assert_eq!(b, c);
        assert_eq!(a, d);
    }

    #[test]
    fn test_all_pair_distances() {

        let m = mat![1.0, 2.0; 5.0, 12.0; 13.0, 27.0];
        let r = all_pair_distances(&m);

        assert_eq!(r.rows(), m.rows());
        assert_eq!(r.cols(), m.rows());
        assert_eq!(*r.get(0, 0).unwrap(), 0.0);
        assert_eq!(*r.get(1, 1).unwrap(), 0.0);
        assert_eq!(*r.get(2, 2).unwrap(), 0.0);

        assert!(*r.get(0, 1).unwrap() - 10.770 <= 0.001);
        assert!(*r.get(0, 2).unwrap() - 27.731 <= 0.001);
        assert!(*r.get(1, 0).unwrap() - 10.770 <= 0.001);
        assert!(*r.get(2, 0).unwrap() - 27.731 <= 0.001);

        assert!(*r.get(1, 2).unwrap() - 17.0 <= 0.001);
        assert!(*r.get(2, 1).unwrap() - 17.0 <= 0.001);
    }
}

