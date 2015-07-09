extern crate libc;
extern crate num;

use self::libc::{c_int, c_float, c_double};
use matrix::Matrix;
use blas::{cblas_saxpy, cblas_daxpy};

// ----------------------------------------------------------------------------

pub trait MatrixScalarOps<T> {
    /// Adds a scalar to each element of the matrix.
    fn add_scalar(&self, scalar: T) -> Matrix<T>;

    /// Multiplies each element of the matrix with a scalar.
    fn mul_scalar(&self, scalar: T) -> Matrix<T>;
}

macro_rules! matrix_scalar_ops_impl {
    ($($t:ty)*) => ($(

        impl MatrixScalarOps<$t> for Matrix<$t> {

            fn add_scalar(&self, scalar: $t) -> Matrix<$t> {

                Matrix::from_vec(
                    self.values().map(|&x| x + scalar).collect(),
                    self.rows(),
                    self.cols()
                ).unwrap()
            }

            fn mul_scalar(&self, scalar: $t) -> Matrix<$t> {

                Matrix::from_vec(
                    self.values().map(|&x| x * scalar).collect(),
                    self.rows(),
                    self.cols()
                ).unwrap()
            }
        }
    )*)
}

matrix_scalar_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

pub trait VectorScalarOps<T> {
    /// Adds a scalar to each element of the vector.
    fn mul_scalar(&self, scalar: T) -> Vec<T>;

    /// Multiplies each element of the vector with a scalar.
    fn add_scalar(&self, scalar: T) -> Vec<T>;
}

macro_rules! vector_scalar_ops_impl {
    ($($t:ty)*) => ($(

        impl VectorScalarOps<$t> for Vec<$t> {

            fn mul_scalar(&self, scalar: $t) -> Vec<$t> {
                self.iter().map(|&x| x * scalar).collect()
            }

            fn add_scalar(&self, scalar: $t) -> Vec<$t> {
                self.iter().map(|&x| x + scalar).collect()
            }
        }
    )*)
}

vector_scalar_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

/// Trait to add a slice to a vector using the underlying BLAS implementation.
pub trait VectorVectorOps<T> {

    /// Adds the given slice `rhs` inplace.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::ops::VectorVectorOps;
    ///
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.add(&y);
    /// assert_eq!(v, vec![4.0, 10.0]);
    /// ```
    fn add(&mut self, rhs: &[T]);
}

impl VectorVectorOps<f32> for Vec<f32> {

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
            )
        }
    }
}

impl VectorVectorOps<f64> for Vec<f64> {

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
            )
        }
    }
}


// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::Matrix;

    #[test]
    fn test_matrix_ops() {

        let m = mat![
            1.0f32, 2.0; 
            3.0, 4.0; 
            5.0, 6.0; 
            7.0, 8.0
        ];

        let a = m.mul_scalar(2.0);
        assert_eq!(a.buf(), &vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
        let b = m.add_scalar(3.0);
        assert_eq!(b.buf(), &vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_vector_ops() {

        let a = vec![1.0f32, 2.0, 3.0];
        let b = a.mul_scalar(3.0);
        assert_eq!(b, [3.0, 6.0, 9.0]);

        let c = a.add_scalar(3.0);
        assert_eq!(c, [4.0, 5.0, 6.0]);
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

}

