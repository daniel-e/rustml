//! Provides scalar, vector, vector-vector, vector-matrix and matrix-matrix operations.
//!
//! Most of the operations are optimized using the underlying BLAS implementation.
//! For each function it is explicitly documented whether or not BLAS is used.
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
extern crate num;

use self::libc::{c_int, c_float, c_double};

use ops::Functions;
use blas::*;
use matrix::Matrix;

// ----------------------------------------------------------------------------

/// Computes `alpha * x + y` and stores the result in `y`. (optimized via BLAS)
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

/// Computes `alpha * op(A) * op(B) + beta * C` and stores the result in `C`. (optimized via BLAS)
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

    if colsa != rowsb {
        panic!(
            format!("Dimensions for d_gemm do not match: {}x{} * {}x{}", 
            rowsa, colsa, rowsb, colsb
        ));
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
/// result in `y`. (optimized via BLAS)
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

/// Computes the L2 norm (euclidean norm) of a vector. (optimized via BLAS)
///
/// ```
/// # #[macro_use] extern crate rustml;
/// # extern crate num;
/// use num::abs;
/// use rustml::ops_inplace::*;
///
/// # fn main() {
/// let x = [1.0, 2.0, 5.0, 9.0];
/// assert!(abs(d_nrm2(&x) - 10.536) <= 0.001);
/// # }
/// ```
///
pub fn d_nrm2(x: &[f64]) -> f64 {

    if x.len() == 0 {
        return 0.0;
    }

    unsafe {
        cblas_dnrm2(
            x.len() as c_int,
            x.as_ptr() as *const c_double,
            1 as c_int
        ) as f64
    }
}

/// Computes `alpha * x + y` and stores the result in `y`. (optimized via BLAS)
/// 
/// Panics if the dimensions of the vectors do not match.
///
/// ```
/// use rustml::ops_inplace::*;
///
/// # fn main() {
/// let     x = [1.0f32, 2.0, 3.0];
/// let mut y = [4.0f32, 2.0, 9.0];
/// s_axpy(3.0f32, &x, &mut y);
/// assert_eq!(y, [7.0f32, 8.0, 18.0]);
/// # }
/// ```
pub fn s_axpy(alpha: f32, x: &[f32], y: &mut [f32]) {

    if x.len() != y.len() {
        panic!("Dimensions do not match.")
    }

    unsafe {
        cblas_saxpy(
            x.len() as c_int,
            alpha as c_float,
            x.as_ptr() as *const c_float,
            1 as c_int,
            y.as_ptr() as *mut c_float,
            1 as c_int
        );
    }
}

/// Computes `alpha * op(A) * op(B) + beta * C` and stores the result in `C`. (optimized via BLAS)
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
///     1.0f32, 2.0, 3.0;
///     4.0, 5.0, 6.0
/// ];
/// let b = mat![
///     2.0f32, 5.0, 2.0, 3.0;
///     1.0, 9.0, 5.0, 4.0;
///     4.0, 6.0, 8.0, 7.0
/// ];
/// let mut c = mat![
///     3.0f32, 4.0, 1.0, 8.0;
///     4.0, 2.0, 6.0, 6.0
/// ];
/// s_gemm(2.0f32, &a, &b, 3.0f32, &mut c, false, false);
/// assert_eq!(
///     c.buf(), 
///     &vec![41.0f32, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]
/// );
/// # }
/// ```
pub fn s_gemm(alpha: f32, a: &Matrix<f32>, b: &Matrix<f32>, 
              beta: f32, c: &mut Matrix<f32>,
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
        cblas_sgemm(Order::RowMajor, 
            if transa { Transpose::Trans } else { Transpose::NoTrans},
            if transb { Transpose::Trans } else { Transpose::NoTrans},
            m     as c_int,
            n     as c_int,
            k     as c_int,
            alpha     as c_float,
            a.buf().as_ptr()  as *const c_float,
            lda               as c_int,
            b.buf().as_ptr()  as *const c_float,
            ldb               as c_int,
            beta              as c_float,
            c.buf().as_ptr()  as *mut c_float,
            ldc               as c_int
        );
    }
}

/// Computes the L2 norm (euclidean norm) of a vector. (optimized via BLAS)
///
/// ```
/// # #[macro_use] extern crate rustml;
/// # extern crate num;
/// use num::abs;
/// use rustml::ops_inplace::*;
///
/// # fn main() {
/// let x = [1.0f32, 2.0, 5.0, 9.0];
/// assert!(abs(s_nrm2(&x) - 10.536) <= 0.001);
/// # }
/// ```
///
pub fn s_nrm2(x: &[f32]) -> f32 {

    if x.len() == 0 {
        return 0.0;
    }

    unsafe {
        cblas_snrm2(
            x.len() as c_int,
            x.as_ptr() as *const c_float,
            1 as c_int
        ) as f32
    }
}

/// Computes `alpha * A * x + beta * y` or `alpha * A^T * x + beta * y` and stores the
/// result in `y`. (optimized via BLAS)
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
///     1.0f32, 2.0, 3.0; 
///     4.0, 2.0, 5.0
/// ];
/// let x = [2.0f32, 6.0, 3.0];
/// let mut y = [7.0f32, 2.0];
///
/// s_gemv(false, 2.0, &a, &x, 3.0, &mut y);
/// assert_eq!(y, [67.0f32, 76.0]);
/// # }
/// ```
///
pub fn s_gemv(trans: bool, alpha: f32, a: &Matrix<f32>, x: &[f32], beta: f32, y: &mut [f32]) {

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
        cblas_sgemv(
            Order::RowMajor, 
            transpose,
            a.rows() as c_int,
            a.cols() as c_int,
            alpha as c_float,
            a.buf().as_ptr() as *const c_float,
            a.cols() as c_int,
            x.as_ptr() as *const c_float,
            1 as c_int,
            beta as c_float,
            y.as_ptr() as *mut c_float,
            1 as c_int
        );
    }
}

// ----------------------------------------------------------------------------

/// Trait for common mathematical functions for scalars, vectors and matrices.
pub trait FunctionsInPlace {

    /// Computes the sigmoid function (i.e. 1/(1+exp(-x))) for a scalar or each
    /// element in a vector or matrix.
    fn isigmoid(&mut self);

    /// Computes the derivative of the sigmoid function (i.e. sigmoid(x) * (1 - sigmoid(x)))
    /// for a scalar or each element in a vector or matrix.
    fn isigmoid_derivative(&mut self);

    /// Takes the reciprocal.
    fn irecip(&mut self);
}

macro_rules! impl_functions_ops_inplace {
    ( $( $x:ty )+ ) => ($(

        impl FunctionsInPlace for $x {

            fn isigmoid(&mut self) {
                *self = 1.0 / (1.0 + (- *self).exp());
            }

            fn isigmoid_derivative(&mut self) {
                *self = self.sigmoid() * (1.0 - self.sigmoid());
            }

            fn irecip(&mut self) {
                *self = self.recip();
            }
        }
    )*)
}

impl_functions_ops_inplace!{ f32 f64 }

impl <T: FunctionsInPlace> FunctionsInPlace for Vec<T> {

    fn isigmoid(&mut self) { self[..].isigmoid(); }
    fn isigmoid_derivative(&mut self) { self[..].isigmoid_derivative(); }
    fn irecip(&mut self) { self[..].irecip(); }
}

impl <T: FunctionsInPlace> FunctionsInPlace for [T] {

    fn isigmoid(&mut self) {

        for i in self {
            i.isigmoid();
        }
    }

    fn isigmoid_derivative(&mut self) {

        for i in self {
            i.isigmoid_derivative();
        }
    }

    fn irecip(&mut self) {
        for i in self {
            i.irecip();
        }
    }
}

impl <T: FunctionsInPlace + Clone> FunctionsInPlace for Matrix<T> {

    fn isigmoid(&mut self) {

        for i in self.iter_mut() {
            i.isigmoid();
        }
    }

    fn isigmoid_derivative(&mut self) {

        for i in self.iter_mut() {
            i.isigmoid_derivative();
        }
    }

    fn irecip(&mut self) {
        for i in self.iter_mut() {
            i.irecip();
        }
    }
}

// ----------------------------------------------------------------------------

/// Trait for matrix-scalar operations.
pub trait MatrixScalarOpsInPlace<T> {

    /// Divides each element of the matrix by the scalar `val`.
    ///
    /// Not accelerated via BLAS.
    fn idiv_scalar(&mut self, val: T);

    /// Multiplies each element of the matrix with the scalar `val`.
    ///
    /// Not accelerated via BLAS.
    fn imul_scalar(&mut self, val: T);

    /// Adds the scalar `val` to each element of the matrix.
    ///
    /// Not accelerated via BLAS.
    fn iadd_scalar(&mut self, val: T);

    /// Subtracts the scalar `val` from each element of the matrix.
    ///
    /// Not accelerated via BLAS.
    fn isub_scalar(&mut self, val: T);
}

macro_rules! impl_matrix_scalar_ops_inplace {
    ( $( $x:ty )+ ) => ($(

        impl MatrixScalarOpsInPlace<$x> for Matrix<$x> {

            fn idiv_scalar(&mut self, val: $x) {
                for i in self.iter_mut() {
                    *i = *i / val;
                }
            }

            fn imul_scalar(&mut self, val: $x) {
                for i in self.iter_mut() {
                    *i = *i * val;
                }
            }

            fn iadd_scalar(&mut self, val: $x) {
                for i in self.iter_mut() {
                    *i = *i + val;
                }
            }

            fn isub_scalar(&mut self, val: $x) {
                for i in self.iter_mut() {
                    *i = *i - val;
                }
            }
        }
    )*)
}

impl_matrix_scalar_ops_inplace!{ f32 }
impl_matrix_scalar_ops_inplace!{ f64 }

// ----------------------------------------------------------------------------

/// Trait for matrix-matrix operations.
pub trait MatrixMatrixOpsInPlace<T> {

    /// Adds the matrix `rhs` to this matrix inplace.
    ///
    /// Implementation details: iterates through the rows of both matrices and
    /// uses `iadd` for the vectors (using BLAS).
    fn iadd(&mut self, rhs: &Matrix<T>);

    /// Subtracts the matrix `rhs` from this matrix inplace.
    ///
    /// Implementation details: iterates through the rows of both matrices and
    /// uses `isub` for the vectors (using BLAS).
    fn isub(&mut self, rhs: &Matrix<T>);

    /// Element wise matrix multiplication.
    ///
    /// (not accelerated via BLAS)
    fn imule(&mut self, rhs: &Matrix<T>);
}

macro_rules! impl_matrix_matrix_ops_inplace {
    ( $( $x:ty, $gemm:expr )+ ) => ($(

        impl MatrixMatrixOpsInPlace<$x> for Matrix<$x> {

            fn iadd(&mut self, rhs: &Matrix<$x>) { 

                assert!(self.rows() == rhs.rows() && self.cols() == rhs.cols(), "Dimensions mismatch.");
                for i in 0..self.rows() {
                    self.row_mut(i).unwrap().iadd(&rhs.row(i).unwrap());
                }/*
                for (i, j) in self.iter_mut().zip(rhs.iter()) {
                    *i = *i + j;
                }*/
            }

            fn isub(&mut self, rhs: &Matrix<$x>) { 

                assert!(self.rows() == rhs.rows() && self.cols() == rhs.cols(), "Dimensions mismatch.");
                for i in 0..self.rows() {
                    self.row_mut(i).unwrap().isub(&rhs.row(i).unwrap());
                }/*
                for (i, j) in self.iter_mut().zip(rhs.iter()) {
                    *i = *i + j;
                }*/
            }

            fn imule(&mut self, rhs: &Matrix<$x>) {

                assert!(self.rows() == rhs.rows() && self.cols() == rhs.cols(), "Dimensions mismatch.");
                for (i, j) in self.iter_mut().zip(rhs.iter()) {
                    *i = *i * j;
                }
            }
        }
    )*)
}

impl_matrix_matrix_ops_inplace!{ f32, s_gemm }
impl_matrix_matrix_ops_inplace!{ f64, d_gemm }

// ----------------------------------------------------------------------------

/// Trait for inplace vector-vector operations.
pub trait VectorVectorOpsInPlace<T> {

    /// Adds the given vector `rhs` to self inplace.
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

    /// Substracts the given vector `rhs` from self inplace.
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

    /// Computes the L2 norm (i.e. the euclidean norm) of the vector.
    ///
    /// # Implementation details
    ///
    /// This operation is optimized via BLAS.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate num;
    /// # extern crate rustml;
    ///
    /// # use num::abs;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let v = vec![3.0, 2.0, 4.0];
    /// assert!(num::abs(v.nrm2() - 5.3852) <= 0.0001);
    /// # }
    /// ```
    fn nrm2(&self) -> T;
}

macro_rules! impl_vector_vector_ops_inplace {
    ( $( $x:ty, $axpy:expr, $nrm:expr )+ ) => ($(

        impl VectorVectorOpsInPlace<$x> for Vec<$x> {
            fn iadd(&mut self, rhs: &[$x]) { (self[..]).iadd(rhs); }
            fn isub(&mut self, rhs: &[$x]) { (self[..]).isub(rhs); }
            fn imul(&mut self, rhs: &[$x]) { (self[..]).imul(rhs); }
            fn idiv(&mut self, rhs: &[$x]) { (self[..]).idiv(rhs); }
            fn nrm2(&self) -> $x { (self[..]).nrm2() }
        }

        impl VectorVectorOpsInPlace<$x> for [$x] {

            fn iadd(&mut self, rhs: &[$x]) {

                assert!(self.len() == rhs.len(), "Dimensions do not match.");
                $axpy(1.0, rhs, self);
            }

            fn isub(&mut self, rhs: &[$x]) {

                assert!(self.len() == rhs.len(), "Dimensions do not match.");
                $axpy(-1.0, rhs, self);
            }

            fn idiv(&mut self, rhs: &[$x]) {

                assert!(self.len() == rhs.len(), "Dimensions do not match.");
                for (a, b) in self.iter_mut().zip(rhs.iter()) {
                    *a /= *b;
                }
            }

            fn imul(&mut self, rhs: &[$x]) {

                assert!(self.len() == rhs.len(), "Dimensions do not match.");
                for (a, b) in self.iter_mut().zip(rhs.iter()) {
                    *a *= *b;
                }
            }

            fn nrm2(&self) -> $x { $nrm(self) }
        }
    )*)
}

impl_vector_vector_ops_inplace!{ f32, s_axpy, s_nrm2 }
impl_vector_vector_ops_inplace!{ f64, d_axpy, d_nrm2 }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate num;
    extern crate time;

    use super::*;
    use matrix::*;
    use self::num::abs;

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
    fn test_s_axpy() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [4.0f32, 2.0, 9.0];
        s_axpy(3.0f32, &x, &mut y);
        assert_eq!(y, [7.0f32, 8.0, 18.0]);
        assert_eq!(x, [1.0f32, 2.0, 3.0]);
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
    fn test_s_gemm() {

        let a = mat![
            1.0f32, 2.0, 3.0;
            4.0, 5.0, 6.0
        ];
        let b = mat![
            2.0f32, 5.0, 2.0, 3.0;
            1.0, 9.0, 5.0, 4.0;
            4.0, 6.0, 8.0, 7.0
        ];
        let mut c = mat![
            3.0f32, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        s_gemm(2.0f32, &a, &b, 3.0f32, &mut c, false, false);
        assert_eq!(c.buf(), &vec![41.0f32, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);
        assert_eq!(a.buf(), &vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // ----

        let aa = mat![
            1.0f32, 4.0;
            2.0, 5.0;
            3.0, 6.0
        ];
        c = mat![
            3.0f32, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        s_gemm(2.0f32, &aa, &b, 3.0f32, &mut c, true, false);
        assert_eq!(c.buf(), &vec![41.0f32, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);
        assert_eq!(aa.buf(), &vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // ----

        let bb = mat![
            2.0f32, 1.0, 4.0;
            5.0, 9.0, 6.0;
            2.0, 5.0, 8.0;
            3.0, 4.0, 7.0
        ];
        c = mat![
            3.0f32, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        s_gemm(2.0f32, &a, &bb, 3.0f32, &mut c, false, true);
        assert_eq!(c.buf(), &vec![41.0f32, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);

        // ----
        c = mat![
            3.0f32, 4.0, 1.0, 8.0;
            4.0, 2.0, 6.0, 6.0
        ];
        s_gemm(2.0f32, &aa, &bb, 3.0f32, &mut c, true, true);
        assert_eq!(c.buf(), &vec![41.0f32, 94.0, 75.0, 88.0, 86.0, 208.0, 180.0, 166.0]);
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

    #[test]
    fn test_s_gemv() {
        let a = mat![
            1.0f32, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let x = [2.0f32, 6.0, 3.0];
        let mut y = [7.0f32, 2.0];

        s_gemv(false, 2.0f32, &a, &x, 3.0f32, &mut y);
        assert_eq!(y, [67.0f32, 76.0]);

        let aa = mat![
            1.0f32, 4.0;
            2.0, 2.0;
            3.0, 5.0
        ];
        y = [7.0f32, 2.0];
        s_gemv(true, 2.0f32, &aa, &x, 3.0f32, &mut y);
        assert_eq!(y, [67.0f32, 76.0]);
    }


    #[test]
    fn test_d_nrm2() {
        let x = [1.0, 2.0, 5.0, 9.0];
        assert!(abs(d_nrm2(&x) - 10.536) <= 0.001);
    }

    #[test]
    fn test_s_nrm2() {
        let x = [1.0f32, 2.0, 5.0, 9.0];
        assert!(abs(s_nrm2(&x) - 10.536) <= 0.001);
    }

    #[test]
    fn test_isigmoid() {

        let mut i = 1.0;
        i.isigmoid();
        assert!(num::abs(i - 0.73106) <= 0.00001);

        i = 1.0;
        i.isigmoid_derivative();
        assert!(num::abs(i - 0.19661) <= 0.00001);

        let mut a = [1.0, 2.0, 3.0];
        a.isigmoid();
        assert!(a.similar(&[0.73106, 0.88080, 0.95257], 0.00001));

        let mut b = [1.0, 2.0];
        b.isigmoid_derivative();
        assert!(b.similar(&vec![0.19661, 0.10499], 0.00002));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_iadd() {

        let mut a = mat![1.0, 2.0, 3.0; 4.0, 1.0, 7.0];
        let b = mat![2.0, 5.0, 9.0; 1.0, 7.0, 3.0];
        a.iadd(&b);
        assert!(a.eq(&mat![3.0, 7.0, 12.0; 5.0, 8.0, 10.0]));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_isub() {

        let mut a = mat![1.0, 2.0, 3.0; 4.0, 1.0, 7.0];
        let     b = mat![2.0, 5.0, 9.0; 1.0, 7.0, 3.0];
        a.isub(&b);
        assert!(a.similar(&mat![-1.0, -3.0, -6.0; 3.0, -6.0, 4.0], 0.0001));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_idiv_scalar() {

        let mut a = mat![
            1.0, 2.0, 3.0;
            4.0, 1.0, 7.0
        ];
        a.idiv_scalar(2.0);
        assert!(a.eq(&mat![
            0.5, 1.0, 1.5;
            2.0, 0.5, 3.5
        ]));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_imul_scalar() {

        let mut a = mat![
            1.0, 2.0, 3.0;
            4.0, 1.0, 7.0
        ];
        a.imul_scalar(2.0);
        assert!(a.eq(&mat![
            2.0, 4.0, 6.0;
            8.0, 2.0, 14.0
        ]));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_iadd_scalar() {

        let mut a = mat![
            1.0, 2.0, 3.0;
            4.0, 1.0, 7.0
        ];
        a.iadd_scalar(2.0);
        assert!(a.eq(&mat![
            3.0, 4.0, 5.0;
            6.0, 3.0, 9.0
        ]));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_isub_scalar() {

        let mut a = mat![
            1.0, 2.0, 3.0;
            4.0, 1.0, 7.0
        ];
        a.isub_scalar(2.0);
        assert!(a.eq(&mat![
            -1.0, 0.0, 1.0;
            2.0, -1.0, 5.0
        ]));
    }

    #[test]
    fn test_matrix_matrix_ops_inplace_imule() {

        let mut a = mat![1.0, 2.0, 3.0; 4.0, 1.0, 7.0];
        let     b = mat![2.0, 5.0, 9.0; 1.0, 7.0, 3.0];
        a.imule(&b);
        assert!(a.similar(&mat![2.0, 10.0, 27.0; 4.0, 7.0, 21.0], 0.0001));
    }

    #[test]
    fn test_recip() {
        let mut a = 2.0;
        a.irecip();
        assert_eq!(a, 0.5);
        let mut b = vec![2.0, 10.0];
        b.irecip();
        assert_eq!(b, vec![0.5, 0.1]);
        let mut c = mat![2.0, 10.0; 4.0, 5.0];
        c.irecip();
        assert_eq!(c, mat![0.5, 0.1; 0.25, 0.2]);
    }
/*
    #[test]
    fn test_iadd_perf() {

        let mut m1 = Matrix::<f64>::random::<f64>(100, 500);
        let m2 = Matrix::<f64>::random::<f64>(100, 500);

        let t1 = time::now();
        for _ in 0..1000 {
            m1.iadd(&m2);
        }
        let t2 = time::now();

        println!("dt = {}", t2 - t1);
    }*/
}

