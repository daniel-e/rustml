//! Functions to compute the distance between vectors.

extern crate libc;

use self::libc::{c_int, c_double, c_float};
use ::matrix::*;
use ::norm::{L2Norm, Norm};
use ::blas::{cblas_daxpy, cblas_saxpy};

/// Computes the distance between two vectors.
pub trait Distance<T> {
    /// Computes the distance between vector `a` and `b` and returns `None`
    /// on failure.
    fn compute(a: &[T], b: &[T]) -> Option<T>;
}

pub struct Euclid;

impl Distance<f64> for Euclid {

    /// Computes the euclidean distance between the vector `a` and `b`.
    ///
    /// Returns `None` if the two vectors have a different length.
    ///
    /// # Implementation details
    ///
    /// First the BLAS function `cblas_daxpy` is used to compute the
    /// difference between the vectors. This requires O(n) additional space
    /// if `n` is the number of elements of each vector. Then, the result
    /// of the L2 norm of the difference is returned.
    fn compute(a: &[f64], b: &[f64]) -> Option<f64> {

        // TODO handling of NaN and stuff like this
        if a.len() != b.len() {
            return None;
        }

        // c = b.clone() does not work here because cblas_daxpy
        // modifies the content of c and cloned() on a slice does
        // not create a copy.
        let c: Vec<f64> = b.to_vec();

        unsafe {
            cblas_daxpy(
                a.len()     as c_int,
                -1.0        as c_double,
                a.as_ptr()  as *const c_double,
                1           as c_int,
                c.as_ptr()  as *mut c_double,
                1           as c_int
            );
        }
        Some(L2Norm::compute(&c))
    }
}

impl Distance<f32> for Euclid {

    /// Computes the euclidean distance between the vector `a` and `b`.
    ///
    /// Returns `None` if the two vectors have a different length.
    ///
    /// # Implementation details
    ///
    /// First the BLAS function `cblas_daxpy` is used to compute the
    /// difference between the vectors. This requires O(n) additional space
    /// if `n` is the number of elements of each vector. Then, the result
    /// of the L2 norm of the difference is returned.
    fn compute(a: &[f32], b: &[f32]) -> Option<f32> {

        // TODO handling of NaN and stuff like this
        if a.len() != b.len() {
            return None;
        }

        // c = b.clone() does not work here because cblas_daxpy
        // modifies the content of c and cloned() on a slice does
        // not create a copy.
        let c: Vec<f32> = b.to_vec();

        unsafe {
            cblas_saxpy(
                a.len()     as c_int,
                -1.0        as c_float,
                a.as_ptr()  as *const c_float,
                1           as c_int,
                c.as_ptr()  as *mut c_float,
                1           as c_int
            );
        }
        Some(L2Norm::compute(&c))
    }
}

pub fn all_pair_distances(m: &Matrix<f64>) -> Matrix<f64> {

    let mut r = Matrix::fill(0.0, m.rows(), m.rows());

    // TODO handling of NaN and stuff like this
    for (i, row1) in m.row_iter().enumerate() {
        for (j, row2) in m.row_iter_at(i + 1).enumerate() {
            let p = j + i + 1;
            let d = Euclid::compute(row1, row2).unwrap();
            r.set(i, p, d);
            r.set(p, i, d);
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

