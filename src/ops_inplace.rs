//! Provides scalar, vector, vector-vector, vector-matrix and matrix-matrix operations.
//!
//! Most of the operations are accelerated using the underlying BLAS implementation.
//! If BLAS is not used for an operation this is explicitly documented.
//!
//! # Examples
//! 
//! The following example adds two vectors using BLAS and stores the result in the first
//! vector.
//!
//! ```
//! use rustml::*;
//!
//! let mut a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let b = [3.0, 6.0, 2.0, 3.0, 6.0, 9.0];
//! a.iadd(&b);
//! assert_eq!(a, [4.0, 8.0, 5.0, 7.0, 11.0, 15.0]);
//! ```
extern crate libc;

use self::libc::{c_int, c_float, c_double};

use blas::*;
use matrix::Matrix;

// ----------------------------------------------------------------------------

/// Computes `alpha * x + y` and stores the result in `y`. (accelerated via BLAS)
/// 
/// Panics if the dimensions of the vectors do not match.
///
/// ```
/// use rustml::ops_inplace::*;
///
/// # fn main() {
/// let     x = [1.0, 2.0, 3.0];
/// let mut y = [4.0, 2.0, 9.0];
/// d_axpy(3.0, &x, &mut y);
/// assert_eq!(y, [7.0, 8.0, 18.0]);
/// # }
/// ```
pub fn d_axpy(alpha: f64, x: &[f64], y: &mut [f64]) {

    if x.len() != y.len() {
        panic!("Dimensions do not match.")
    }

    unsafe {
        cblas_daxpy(
            x.len() as c_int,
            alpha as c_double,
            x.as_ptr() as *const c_double,
            1 as c_int,
            y.as_ptr() as *mut c_double,
            1 as c_int
        );
    }
}

/// Computes `alpha * op(A) * op(B) + beta * C` and stores the result in `C`. (accelerated via BLAS)
///
/// If `transa` is `true` the function `op(A)` returns the transpose of `A`,
/// otherwise `A` is returned. If `transb` is `true` the function `op(B)` returns the
/// transpose of `B`, otherwise `B` is returned.
///
/// Panics if the dimensions of the matrices do not match.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::ops_inplace::*;
/// use rustml::matrix::*;
///
/// # fn main() {
/// let a = mat![
///     1.0, 2.0, 3.0;
///     4.0, 5.0, 6.0
/// ];
/// let b = mat![
///     2.0, 5.0, 2.0, 3.0;
///     1.0, 9.0, 5.0, 4.0;
///     4.0, 6.0, 8.0, 7.0
/// ];
/// let mut c = mat![
///     3.0, 4.0, 1.0, 8.0;
///     4.0, 2.0, 6.0, 6.0
/// ];
/// d_gemm(2.0, &a, &b, 3.0, &mut c, false, false);
/// assert_eq!(
///     c.buf(), 
///     &vec![41.0, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]
/// );
/// # }
/// ```
pub fn d_gemm(alpha: f64, a: &Matrix<f64>, b: &Matrix<f64>, 
              beta: f64, c: &mut Matrix<f64>,
              transa: bool, transb: bool) {

    let rowsa = if transa { a.cols() } else { a.rows() };
    let colsa = if transa { a.rows() } else { a.cols() };
    let rowsb = if transb { b.cols() } else { b.rows() };
    let colsb = if transb { b.rows() } else { b.cols() };

    if colsa != rowsb || rowsa != c.rows() || colsb != c.cols() {
        panic!("Dimensions do not match.");
    }

    let m = c.rows();
    let n = c.cols();
    let k = colsa;

    let lda = if !transa { k } else { m };
    let ldb = if !transb { n } else { k };
    let ldc = c.cols();

    unsafe {
        cblas_dgemm(Order::RowMajor, 
            if transa { Transpose::Trans } else { Transpose::NoTrans},
            if transb { Transpose::Trans } else { Transpose::NoTrans},
            m     as c_int,
            n     as c_int,
            k     as c_int,
            alpha     as c_double,
            a.buf().as_ptr()  as *const c_double,
            lda               as c_int,
            b.buf().as_ptr()  as *const c_double,
            ldb               as c_int,
            beta              as c_double,
            c.buf().as_ptr()  as *mut c_double,
            ldc               as c_int
        );
    }
}

/// Computes `alpha * A * x + beta * y` or `alpha * A^T * x + beta * y` and stores the
/// result in `y`. (accelerated via BLAS)
///
/// If `trans` is `true` the transpose of `A` is used.
///
/// Panics if the dimensions of the matrix and the vector do not match.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::ops_inplace::*;
/// use rustml::matrix::*;
///
/// # fn main() {
/// let a = mat![
///     1.0, 2.0, 3.0; 
///     4.0, 2.0, 5.0
/// ];
/// let x = [2.0, 6.0, 3.0];
/// let mut y = [7.0, 2.0];
///
/// d_gemv(false, 2.0, &a, &x, 3.0, &mut y);
/// assert_eq!(y, [67.0, 76.0]);
/// # }
/// ```
///
pub fn d_gemv(trans: bool, alpha: f64, a: &Matrix<f64>, x: &[f64], beta: f64, y: &mut [f64]) {

    if !trans {
        if a.cols() != x.len() || a.rows() != y.len() {
            panic!("Invalid dimensions.");
        }
    } else {
        if a.rows() != x.len() || a.cols() != y.len() {
            panic!("Invalid dimensions.");
        }
    }

    let transpose = if trans { Transpose::Trans } else { Transpose::NoTrans };

    unsafe {
        cblas_dgemv(
            Order::RowMajor, 
            transpose,
            a.rows() as c_int,
            a.cols() as c_int,
            alpha as c_double,
            a.buf().as_ptr() as *const c_double,
            a.cols() as c_int,
            x.as_ptr() as *const c_double,
            1 as c_int,
            beta as c_double,  // beta
            y.as_ptr() as *mut c_double,
            1 as c_int
        );
    }
}

// ----------------------------------------------------------------------------

/// Trait for inplace vector-vector operations.
pub trait VectorVectorOpsInPlace<T> {

    /// Adds the given slice `rhs` to self inplace.
    ///
    /// This operation uses BLAS for high performance for vectors with elements
    /// of type f32 and f64. Panics if the dimensions of the vectors do not match.
    /// 
    /// # Example
    ///
    /// ```
    /// use rustml::*;
    ///
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.iadd(&y);
    /// assert_eq!(v, vec![4.0, 10.0]);
    /// ```
    fn iadd(&mut self, rhs: &[T]);

    /// Substracts the given slice `rhs` from self inplace.
    ///
    /// This operation uses BLAS for high performance for vectors with elements
    /// of type f32 and f64. Panics if the dimensions of the vectors do not match.
    /// 
    /// # Example
    ///
    /// ```
    /// use rustml::*;
    ///
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.isub(&y);
    /// assert_eq!(v, vec![-2.0, -6.0]);
    /// ```
    fn isub(&mut self, rhs: &[T]);

    /// Computes an element-wise vector-vector multiplication and stores the result
    /// in self.
    ///
    /// Panics if the dimensions of the vectors do not match.
    ///
    /// # Implementation details
    ///
    /// This operation does not use BLAS. Instead a simple loop is used to compute
    /// the element-wise vector-vector multiplication.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::*;
    ///
    /// let mut v = vec![3.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.imul(&y);
    /// assert_eq!(v, vec![9.0, 16.0]);
    /// ```
    fn imul(&mut self, rhs: &[T]);

    /// Computes an element-wise vector-vector division and stores the result
    /// in self.
    ///
    /// For all elements in `self` the element at index `i` in self is divided
    /// by the element in `rhs` at the same index.
    ///
    /// Panics if the dimensions of the vectors do not match.
    ///
    /// # Implementation details
    ///
    /// This operation does not use BLAS. Instead a simple loop is used to compute
    /// the element-wise vector-vector division.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::*;
    ///
    /// let mut v = vec![3.0, 2.0];
    /// let y = vec![2.0, 8.0];
    /// v.idiv(&y);
    /// assert_eq!(v, vec![1.5, 0.25]);
    /// ```
    fn idiv(&mut self, rhs: &[T]);
}

macro_rules! impl_vector_vector_ops_inplace {
    ( $( $x:ty, $y:expr, $z:ty )+ ) => ($(

        impl VectorVectorOpsInPlace<$x> for Vec<$x> {
            fn iadd(&mut self, rhs: &[$x]) { (self[..]).iadd(rhs); }
            fn isub(&mut self, rhs: &[$x]) { (self[..]).isub(rhs); }
            fn imul(&mut self, rhs: &[$x]) { (self[..]).imul(rhs); }
            fn idiv(&mut self, rhs: &[$x]) { (self[..]).idiv(rhs); }
        }

        impl VectorVectorOpsInPlace<$x> for [$x] {

            fn iadd(&mut self, rhs: &[$x]) {

                if self.len() != rhs.len() {
                    panic!("Vectors must have the same length.");
                }

                unsafe {
                    $y(
                        self.len()    as c_int,
                        1.0           as $z,
                        rhs.as_ptr()  as *const $z,
                        1             as c_int,
                        self.as_ptr() as *mut $z,
                        1             as c_int
                    )
                }
            }

            fn isub(&mut self, rhs: &[$x]) {

                if self.len() != rhs.len() {
                    panic!("Vectors must have the same length.");
                }

                unsafe {
                    $y(
                        self.len()    as c_int,
                        -1.0          as $z,
                        rhs.as_ptr()  as *const $z,
                        1             as c_int,
                        self.as_ptr() as *mut $z,
                        1             as c_int
                    )
                }
            }

            fn idiv(&mut self, rhs: &[$x]) {

                if self.len() != rhs.len() {
                    panic!("Vectors must have the same length.");
                }

                for (a, b) in self.iter_mut().zip(rhs.iter()) {
                    *a /= *b;
                }
            }

            fn imul(&mut self, rhs: &[$x]) {

                if self.len() != rhs.len() {
                    panic!("Vectors must have the same length.");
                }

                for (a, b) in self.iter_mut().zip(rhs.iter()) {
                    *a *= *b;
                }
            }
        }
    )*)
}

impl_vector_vector_ops_inplace!{ f32, cblas_saxpy, c_float }
impl_vector_vector_ops_inplace!{ f64, cblas_daxpy, c_double }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

    #[test]
    fn test_add_vectorf32() {

        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = vec![2.0, 5.0, 9.0, 15.0];
        x.iadd(&y);

        assert_eq!(x, vec![3.0, 7.0, 12.0, 19.0]);
        assert_eq!(y, vec![2.0, 5.0, 9.0, 15.0]);

        let mut a = [1.0, 2.0, 3.0, 4.0];
        a.iadd(&y);
        assert_eq!(a, [3.0, 7.0, 12.0, 19.0]);
    }

    #[test]
    fn test_add_vectorf64() {

        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![2.0, 5.0, 9.0, 15.0];
        x.iadd(&y);

        assert_eq!(x, vec![3.0, 7.0, 12.0, 19.0]);
        assert_eq!(y, vec![2.0, 5.0, 9.0, 15.0]);

        let mut a = [1.0, 2.0, 3.0, 4.0];
        a.iadd(&y);
        assert_eq!(a, [3.0, 7.0, 12.0, 19.0]);
    }

    #[test]
    fn test_sub_vectorf32() {

        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = vec![2.0, 5.0, 9.0, 15.0];
        x.isub(&y);

        assert_eq!(x, vec![-1.0, -3.0, -6.0, -11.0]);
        assert_eq!(y, vec![2.0, 5.0, 9.0, 15.0]);

        let mut a = [1.0, 2.0, 3.0, 4.0];
        a.isub(&y);
        assert_eq!(a, [-1.0, -3.0, -6.0, -11.0]);
    }

    #[test]
    fn test_sub_vectorf64() {

        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![2.0, 5.0, 9.0, 15.0];
        x.isub(&y);

        assert_eq!(x, vec![-1.0, -3.0, -6.0, -11.0]);
        assert_eq!(y, vec![2.0, 5.0, 9.0, 15.0]);

        let mut a = [1.0, 2.0, 3.0, 4.0];
        a.isub(&y);
        assert_eq!(a, [-1.0, -3.0, -6.0, -11.0]);
    }

    #[test]
    fn test_d_axpy() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 2.0, 9.0];
        d_axpy(3.0, &x, &mut y);
        assert_eq!(y, [7.0, 8.0, 18.0]);
        assert_eq!(x, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_d_gemm() {

        let a = mat![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0
        ];
        let b = mat![
            2.0, 5.0, 2.0, 3.0;
            1.0, 9.0, 5.0, 4.0;
            4.0, 6.0, 8.0, 7.0
        ];
        let mut c = mat![
            3.0, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        d_gemm(2.0, &a, &b, 3.0, &mut c, false, false);
        assert_eq!(c.buf(), &vec![41.0, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);
        assert_eq!(a.buf(), &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // ----

        let aa = mat![
            1.0, 4.0;
            2.0, 5.0;
            3.0, 6.0
        ];
        c = mat![
            3.0, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        d_gemm(2.0, &aa, &b, 3.0, &mut c, true, false);
        assert_eq!(c.buf(), &vec![41.0, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);
        assert_eq!(aa.buf(), &vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // ----

        let bb = mat![
            2.0, 1.0, 4.0;
            5.0, 9.0, 6.0;
            2.0, 5.0, 8.0;
            3.0, 4.0, 7.0
        ];
        c = mat![
            3.0, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        d_gemm(2.0, &a, &bb, 3.0, &mut c, false, true);
        assert_eq!(c.buf(), &vec![41.0, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);

        // ----
        c = mat![
            3.0, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        d_gemm(2.0, &aa, &bb, 3.0, &mut c, true, true);
        assert_eq!(c.buf(), &vec![41.0, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);
    }

    #[test]
    fn test_d_gemv() {
        let a = mat![
            1.0, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let x = [2.0, 6.0, 3.0];
        let mut y = [7.0, 2.0];

        d_gemv(false, 2.0, &a, &x, 3.0, &mut y);
        assert_eq!(y, [67.0, 76.0]);

        let aa = mat![
            1.0, 4.0;
            2.0, 2.0;
            3.0, 5.0
        ];
        y = [7.0, 2.0];
        d_gemv(true, 2.0, &aa, &x, 3.0, &mut y);
        assert_eq!(y, [67.0, 76.0]);
    }
}

