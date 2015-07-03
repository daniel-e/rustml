//! Functions to operate on vectors.
extern crate libc;
extern crate num;

use std::cmp::{PartialEq, min};
use self::libc::{c_void, size_t, c_int, c_float, c_double};
use std::mem;
use std::marker::Copy;
use self::num::traits::Num;

use blas::{cblas_saxpy, cblas_daxpy};

/// Trait to add a slice to a vector using the underlying BLAS implementation.
pub trait AddVector<T: Num> {

    /// Adds the given slice `rhs` inplace.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// assert_eq!(v, vec![4.0, 10.0]);
    /// ```
    fn add(&mut self, rhs: &[T]);
}

impl AddVector<f32> for Vec<f32> {

    fn add(&mut self, rhs: &[f32]) {

        if self.len() != rhs.len() {
            panic!("Vectors must have the same length.");
        }

        unsafe {
            cblas_saxpy(
                self.len()    as c_int,
                1.0           as c_float,
                rhs.as_ptr()  as *const c_float,
                1             as c_int,
                self.as_ptr() as *mut c_float,
                1             as c_int
            );
        }
    }
}

impl AddVector<f64> for Vec<f64> {

    /// Adds the given slice `rhs` to the current vector.
    ///
    /// Panics if vector and slice have different lengths.
    fn add(&mut self, rhs: &[f64]) {

        if self.len() != rhs.len() {
            panic!("Vectors must have the same length.");
        }

        unsafe {
            cblas_daxpy(
                self.len()    as c_int,
                1.0           as c_double,
                rhs.as_ptr()  as *const c_double,
                1             as c_int,
                self.as_ptr() as *mut c_double,
                1             as c_int
            );
        }
    }
}

// ------------------------------------------------------------------

/// Groups equal elements into one element and counts them.
pub fn group<T: PartialEq + Clone>(v: &Vec<T>) -> Vec<(T, usize)> {

    let mut r: Vec<(T, usize)> = Vec::new();
    for val in v {
        if r.len() == 0 {
            r.push((val.clone(), 1));
        } else {
            let mut x = r.pop().unwrap();
            if x.0 != *val {
                r.push(x);
                x = (val.clone(), 0);
            }
            x.1 += 1;
            r.push(x);
        }
    }
    r
}


extern {
    fn memcpy(dst: *mut c_void, src: *const c_void, n: size_t);
}

/// Copies elements from `src` to `dst`.
///
/// # Implementation details
///
/// This function uses the C function call `memcpy` to copy the memory.
pub fn copy_memory<T: Copy>(dst: &mut [T], src: &[T], n: usize) -> usize {

    let c = min(min(dst.len(), src.len()), n);
    unsafe {
        memcpy(
            dst.as_ptr()              as *mut c_void, 
            src.as_ptr()              as *const c_void,
            (c * mem::size_of::<T>()) as size_t
        );
    }
    c
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vectorf32() {

        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = vec![2.0, 5.0, 9.0, 15.0];
        x.add(&y);

        assert_eq!(x, vec![3.0, 7.0, 12.0, 19.0]);
        assert_eq!(y, vec![2.0, 5.0, 9.0, 15.0]);
    }

    #[test]
    fn test_add_vectorf64() {

        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![2.0, 5.0, 9.0, 15.0];
        x.add(&y);

        assert_eq!(x, vec![3.0, 7.0, 12.0, 19.0]);
        assert_eq!(y, vec![2.0, 5.0, 9.0, 15.0]);
    }

    #[test]
    fn test_group() {

        let mut v = vec![1.0, 1.0, 2.0, 7.0, 7.0, 9.0, 9.0, 9.0];
        let mut r = group(&v);
        assert_eq!(r, vec![(1.0, 2), (2.0, 1), (7.0, 2), (9.0, 3)]);

        v = vec![];
        r = group(&v);
        assert_eq!(r, vec![]);

        v = vec![1.0, 2.0, 2.0, 2.0, 3.0, 4.0];
        r = group(&v);
        assert_eq!(r, vec![(1.0, 1), (2.0, 3), (3.0, 1), (4.0, 1)]);
    }

    #[test]
    fn test_copy_memory() {

        let mut a = [0, 0, 0, 0];
        let b = [1, 2, 3, 4];
        assert_eq!(copy_memory(&mut a, &b, 4), 4);
        assert_eq!(a, b);

        assert_eq!(copy_memory(&mut a, &b, 5), 4);

        let mut c = [1.0, 2.0, 3.0];
        let d = [5.0, 6.0, 7.9];
        assert_eq!(copy_memory(&mut c, &d, 3), 3);
        assert_eq!(c, d);
    }
}

