//! Functions to operate on vectors.
extern crate libc;
extern crate num;
extern crate rand;

use std::cmp::{PartialEq, min};
use self::libc::{c_void, size_t, c_int, c_float, c_double};
use std::mem;
use std::marker::Copy;
use self::num::traits::{Float, Num};
use std::iter;
use self::rand::{thread_rng, Rng, Rand};

use blas::{cblas_saxpy, cblas_daxpy};

/// Trait to add a slice to a vector using the underlying BLAS implementation.
pub trait AddVector<T> {

    /// Adds the given slice `rhs` inplace.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::vectors::AddVector;
    ///
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.add(&y);
    /// assert_eq!(v, vec![4.0, 10.0]);
    /// ```
    fn add(&mut self, rhs: &[T]);
}

impl <T: Float> AddVector<T> for Vec<T> {

    fn add(&mut self, rhs: &[T]) {

        if self.len() != rhs.len() {
            panic!("Vectors must have the same length.");
        }

        unsafe {
            match mem::size_of::<T>() {
                4 => cblas_saxpy(
                    self.len()    as c_int,
                    1.0           as c_float,
                    rhs.as_ptr()  as *const c_float,
                    1             as c_int,
                    self.as_ptr() as *mut c_float,
                    1             as c_int
                ),
                8 => cblas_daxpy(
                    self.len()    as c_int,
                    1.0           as c_double,
                    rhs.as_ptr()  as *const c_double,
                    1             as c_int,
                    self.as_ptr() as *mut c_double,
                    1             as c_int
                ),
                _ => panic!("size")
            }
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

// ------------------------------------------------------------------

/// Creates a vector for which all elements are equal to zero.
///
/// # Example
///
/// ```
/// use rustml::vectors::zero;
///
/// let b = zero::<i32>(5);
/// assert_eq!(b, vec![0, 0, 0, 0, 0]);
/// let c = zero::<i32>(0);
/// assert_eq!(c, vec![]);
/// ```
pub fn zero<T: Num + Clone>(n: usize) -> Vec<T> {

    // TODO more efficient implementation
    iter::repeat(T::zero()).take(n).collect()
}

// ------------------------------------------------------------------

/// Creates a vector with random elements.
///
/// # Example
///
// TODO
pub fn random<T: Rand + Clone>(n: usize) -> Vec<T> {

    thread_rng().gen_iter::<T>().take(n).collect::<Vec<T>>()
}

// ------------------------------------------------------------------

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
    extern crate num;
    use super::*;

    #[test]
    fn test_zero() {

        let a = zero::<f32>(4);
        assert_eq!(a, vec![0.0, 0.0, 0.0, 0.0]);
        let b = zero::<i32>(5);
        assert_eq!(b, vec![0, 0, 0, 0, 0]);
        let c = zero::<i32>(0);
        assert_eq!(c, vec![]);
    }

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
