extern crate num;
extern crate libc;

use self::libc::{c_int, c_double};
use std::iter::repeat;

use blas::*;
use matrix::Matrix;
use ops_inplace::{VectorVectorOpsInPlace, d_gemv, s_gemv, FunctionsInPlace};
use vectors::zero;

// ----------------------------------------------------------------------------

/// Trait for common mathematical functions for scalars, vectors and matrices.
pub trait Functions {

    /// Computes the sigmoid function (i.e. 1/(1+exp(-x))) for a scalar or each
    /// element in a vector or matrix.
    fn sigmoid(&self) -> Self;

    /// Computes the derivative of the sigmoid function (i.e. sigmoid(1 - sigmoid(x)))
    /// for a scalar or each element in a vector or matrix.
    fn sigmoid_derivative(&self) -> Self;
}

macro_rules! impl_functions {
    ( $( $x:ty )+ ) => ($(

        impl Functions for $x {

            fn sigmoid(&self) -> $x {
                1.0 / (1.0 + (- *self).exp())
            }

            fn sigmoid_derivative(&self) -> $x {
                self.sigmoid() * (1.0 - self.sigmoid())
            }
        }
    )*)
}

impl_functions!{ f32 f64 }

impl <T: Functions + FunctionsInPlace + Clone> Functions for Vec<T> {

    fn sigmoid(&self) -> Self {
        let mut x = self.clone();
        x.isigmoid();
        x
    }

    fn sigmoid_derivative(&self) -> Self { 
        let mut x = self.clone();
        x.isigmoid_derivative();
        x
    }
}

impl <T: Functions + FunctionsInPlace + Clone> Functions for Matrix<T> {

    fn sigmoid(&self) -> Self {
        let mut x = self.clone();
        x.isigmoid();
        x
    }

    fn sigmoid_derivative(&self) -> Self {
        let mut x = self.clone();
        x.isigmoid_derivative();
        x
    }
}

// ----------------------------------------------------------------------------

pub trait VectorOps<T> {

    fn map<F, U>(&self, f: F) -> Vec<U>
        where F: Fn(&T) -> U;
}

pub trait VectorOpsSigned<T> {

    fn abs(&self) -> Vec<T>;
}

macro_rules! vector_ops_impl {
    ($($t:ty)*) => ($(

        impl VectorOps<$t> for Vec<$t> {
            fn map<F, U>(&self, f: F) -> Vec<U> 
                where F: Fn(& $t) -> U {
                let mut v: Vec<U> = Vec::new();
                for i in self.iter() {
                    v.push(f(i));
                }
                v
            }
        }
    )*)
}

vector_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

macro_rules! vector_ops_signed_impl {
    ($($t:ty)*) => ($(

        impl VectorOpsSigned<$t> for Vec<$t> {
            fn abs(&self) -> Vec<$t> {
                self.iter().map(|&x| num::abs(x)).collect()
            }
        }
    )*)
}

vector_ops_signed_impl!{ isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

/// Trait for operations on matrices.
pub trait MatrixOps<T> {

    /// Computes the reciprocal (inverse) of each element of the matrix
    /// and returns the result in a new matrix.
    fn recip(&self) -> Matrix<T>;

    /// Returns the maximum element of the matrix or `None`
    /// if the matrix is empty.
    fn max(&self) -> Option<&T>;
}

macro_rules! matrix_ops_impl {
    ($($t:ty)*) => ($(

        impl MatrixOps<$t> for Matrix<$t> {

            fn recip(&self) -> Matrix<$t> {
                self.map(|&x| (1.0 as $t) / x)
            }

            fn max(&self) -> Option<&$t> {
                match self.empty() {
                    true  => None,
                    false => {
                        let mut val = self.values().next().unwrap();
                        for i in self.values() {
                            if i > val {
                                val = i;
                            }
                        }
                        Some(val)
                    }
                }
            }
        }
    )*)
}

matrix_ops_impl!{ f32 f64 }


// ----------------------------------------------------------------------------

/// Trait for matrix scalar operations.
pub trait MatrixScalarOps<T> {
    /// Adds a scalar to each element of the matrix and returns
    /// the result.
    fn add_scalar(&self, scalar: T) -> Matrix<T>;

    /// Subtracts a scalar from each element of the matrix and returns
    /// the result.
    fn sub_scalar(&self, scalar: T) -> Matrix<T>;

    /// Multiplies each element of the matrix with a scalar
    /// and returns the result.
    fn mul_scalar(&self, scalar: T) -> Matrix<T>;

    /// Divides each element of the matrix by a scalar
    /// and returns the result.
    fn div_scalar(&self, scalar: T) -> Matrix<T>;
}

// ----------------------------------------------------------------------------

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

            fn sub_scalar(&self, scalar: $t) -> Matrix<$t> {

                Matrix::from_vec(
                    self.values().map(|&x| x - scalar).collect(),
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

            fn div_scalar(&self, scalar: $t) -> Matrix<$t> {

                Matrix::from_vec(
                    self.values().map(|&x| x / scalar).collect(),
                    self.rows(),
                    self.cols()
                ).unwrap()
            }
        }
    )*)
}

matrix_scalar_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

/// Trait for vector scalar operations.
pub trait VectorScalarOps<T> {
    /// Multiplies each element of the vector with the scalar and returns
    /// the result.
    fn mul_scalar(&self, scalar: T) -> Vec<T>;

    /// Divides each element of the evector by the scalar and returns
    /// the result.
    fn div_scalar(&self, scalar: T) -> Vec<T>;

    /// Adds a scalar to each element of the vector and returns
    /// the result.
    fn add_scalar(&self, scalar: T) -> Vec<T>;

    /// Subtracts a scalar from each element of the vector 
    /// and returns the result.
    fn sub_scalar(&self, scalar: T) -> Vec<T>;
}

macro_rules! vector_scalar_ops_impl {
    ($($t:ty)*) => ($(

        impl VectorScalarOps<$t> for Vec<$t> {

            fn mul_scalar(&self, scalar: $t) -> Vec<$t> {
                self.iter().map(|&x| x * scalar).collect()
            }

            fn div_scalar(&self, scalar: $t) -> Vec<$t> {
                self.iter().map(|&x| x / scalar).collect()
            }

            fn add_scalar(&self, scalar: $t) -> Vec<$t> {
                self.iter().map(|&x| x + scalar).collect()
            }

            fn sub_scalar(&self, scalar: $t) -> Vec<$t> {
                self.iter().map(|&x| x - scalar).collect()
            }
        }
    )*)
}

vector_scalar_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

/// Trait for vector vector operations.
pub trait VectorVectorOps<T> {

    fn sub(&self, rhs: &[T]) -> Vec<T>;

    fn add(&self, rhs: &[T]) -> Vec<T>;

    /// Element wise multiplication.
    fn mul(&self, rhs: &[T]) -> Vec<T>;

    fn div(&self, rhs: &[T]) -> Vec<T>;

    fn mutate<F>(&self, f: F) -> Vec<T>
        where F: Fn(T) -> T;

    fn col_mul_row(&self, rhs: &[T]) -> Matrix<T>;
}

macro_rules! vector_vector_ops_impl {
    ($($t:ty)*) => ($(

        impl VectorVectorOps<$t> for [$t] {
            fn sub(&self, v: &[$t]) -> Vec<$t> {
                self.iter().zip(v.iter()).map(|(&x, &y)| x - y).collect()
            }

            fn add(&self, v: &[$t]) -> Vec<$t> {
                self.iter().zip(v.iter()).map(|(&x, &y)| x + y).collect()
            }

            fn mul(&self, v: &[$t]) -> Vec<$t> {
                self.iter().zip(v.iter()).map(|(&x, &y)| x * y).collect()
            }

            fn div(&self, v: &[$t]) -> Vec<$t> {
                self.iter().zip(v.iter()).map(|(&x, &y)| x / y).collect()
            }

            fn mutate<F>(&self, f: F) -> Vec<$t>
                where F: Fn($t) -> $t {

                self.iter().map(|&x| f(x)).collect()
            }

            fn col_mul_row(&self, rhs: &[$t]) -> Matrix<$t> {

                let mut m = Matrix::fill(0 as $t, self.len(), rhs.len());
                for r in (0..self.len()) {
                    for c in (0..rhs.len()) {
                        *(m.get_mut(r, c).unwrap()) = self[r] * rhs[c];
                    }
                }
                m
            }
        }

        impl VectorVectorOps<$t> for Vec<$t> {
            fn sub(&self, v: &[$t])                 -> Vec<$t> { (self[..]).sub(v)    }
            fn add(&self, v: &[$t])                 -> Vec<$t> { (self[..]).add(v)    }
            fn mul(&self, v: &[$t])                 -> Vec<$t> { (self[..]).mul(v)    }
            fn div(&self, v: &[$t])                 -> Vec<$t> { (self[..]).div(v)    }
            fn mutate<F: Fn($t) -> $t>(&self, f: F) -> Vec<$t> { (self[..]).mutate(f) }
            fn col_mul_row(&self, v: &[$t])         -> Matrix<$t> { (self[..]).col_mul_row(v) }
        }
    )*)
}

vector_vector_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

/// Trait for matrix vector operations.
pub trait MatrixVectorOps<T> {

    /// Adds the given vector to each row of the matrix.
    fn add_row(&self, rhs: &[T]) -> Matrix<T>;

    /// Subtracts the given vector from each row of the matrix.
    fn sub_row(&self, rhs: &[T]) -> Matrix<T>;

    /// Multiplies the matrix with a vector (i.e. `X*v`)
    /// and returns the result.
    ///
    /// # Implementation details
    ///
    /// This operation is done via the underlying BLAS implementation.
    fn mul_vec(&self, v: &[T]) -> Vec<T>;

    /// Multiplies the transpose of the matrix with a vector (i.e. X<sup>T</sup> * v)
    /// and returns the result.
    ///
    /// # Implementation details
    ///
    /// This operation is done via the underlying BLAS implementation.
    fn transp_mul_vec(&self, v: &[T]) -> Vec<T>;
}

macro_rules! matrix_vector_ops_impl {
    ($($t:ty, $gemv:expr)*) => ($(

        impl MatrixVectorOps<$t> for Matrix<$t> {

            fn add_row(&self, rhs: &[$t]) -> Matrix<$t> {

                let mut m = self.clone();
                for i in (0..m.rows()) {
                    let mut r = m.row_mut(i).unwrap();
                    r.iadd(rhs);
                }
                m
            }

            fn sub_row(&self, rhs: &[$t]) -> Matrix<$t> {

                let mut m = self.clone();
                for i in (0..m.rows()) {
                    let mut r = m.row_mut(i).unwrap();
                    r.isub(rhs);
                }
                m
            }

            fn mul_vec(&self, v: &[$t]) -> Vec<$t> {

                let mut y = zero(self.rows());
                $gemv(false, 1.0, self, v, 0.0, &mut y);
                y
            }

            fn transp_mul_vec(&self, v: &[$t]) -> Vec<$t> {

                let mut y = zero(self.cols());
                $gemv(true, 1.0, self, v, 0.0, &mut y);
                y
            }
        }
    )*)
}

matrix_vector_ops_impl!{ f32, s_gemv }
matrix_vector_ops_impl!{ f64, d_gemv }

// ----------------------------------------------------------------------------

/// Trait for matrix vector multiplication.
pub trait MatrixVectorMul<T> {

    /// Computes Xv-y
    fn mul_vec_minus_vec(&self, v: &[T], y: &[T]) -> Vec<T>;

    /// Computes (alpha * Xv + beta * y) or (alpha * X^T * v + beta * y)
    fn mul_dgemv(&self, trans: bool, alpha: f64, x: &[T], beta: f64, y: &[T]) -> Vec<T>;

    /// Computes (alhpa * Xv) or (alpha * X^T * v)
    fn mul_scalar_vec(&self, trans: bool, alpha: f64, x: &[T]) -> Vec<T>;
}


impl MatrixVectorMul<f64> for Matrix<f64> {

    fn mul_vec_minus_vec(&self, v: &[f64], y: &[f64]) -> Vec<f64> {

        if self.cols() != v.len() || self.rows() != y.len() {
            panic!("Invalid dimensions.");
        }

        // this will be modified by cblas_dgemv
        let targets = y.to_vec();

        unsafe {
            cblas_dgemv(
                Order::RowMajor, 
                Transpose::NoTrans,
                self.rows() as c_int,
                self.cols() as c_int,
                1.0 as c_double,
                self.buf().as_ptr() as *const c_double,
                self.cols() as c_int,
                v.as_ptr() as *const c_double,
                1 as c_int,
                -1.0 as c_double,  // beta
                targets.as_ptr() as *mut c_double,
                1 as c_int
            );
        }
        targets
    }

    fn mul_dgemv(&self, trans: bool, alpha: f64, x: &[f64], beta: f64, y: &[f64]) -> Vec<f64> {

        if !trans {
            if self.cols() != x.len() || self.rows() != y.len() {
                panic!("Invalid dimensions.");
            }
        } else {
            if self.rows() != x.len() || self.cols() != y.len() {
                panic!("Invalid dimensions.");
            }
        }

        let transpose = if trans { Transpose::Trans } else { Transpose::NoTrans };
        // this will be modified by cblas_dgemv
        let r = y.to_vec();

        unsafe {
            cblas_dgemv(
                Order::RowMajor, 
                transpose,
                self.rows() as c_int,
                self.cols() as c_int,
                alpha as c_double,
                self.buf().as_ptr() as *const c_double,
                self.cols() as c_int,
                x.as_ptr() as *const c_double,
                1 as c_int,
                beta as c_double,  // beta
                r.as_ptr() as *mut c_double,
                1 as c_int
            );
        }
        r
    }

    /// Computes (alhpa * Xv) or (alpha * X^T * v)
    fn mul_scalar_vec(&self, trans: bool, alpha: f64, x: &[f64]) -> Vec<f64> {

        let n = if trans { self.cols() } else { self.rows() };
        self.mul_dgemv(
            trans, alpha, x, 0.0,
            &repeat(0.0).take(n).collect::<Vec<f64>>()
        )
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate num;

    use self::num::abs;
    use super::*;
    use matrix::*;
    use math::*;
    use std::f64;

    #[test]
    fn test_matrix_ops() {
        let m = mat![
            1.0f32, 2.0; 
            10.0, 4.0
        ];
        let r = m.recip();
        assert_eq!(r.buf(), &vec![1.0, 0.5, 0.1, 0.25]);
    }

    #[test]
    fn test_matrix_scalar_ops() {

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
        let c = m.sub_scalar(3.0);
        assert_eq!(c.buf(), &vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let d = m.div_scalar(2.0);
        assert_eq!(d.buf(), &vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    }

    #[test]
    fn test_matrix_vector_ops() {
        let m = mat![
            1.0f32, 2.0; 
            3.0, 4.0; 
            5.0, 6.0; 
            7.0, 8.0
        ];

        let a = m.add_row(&[2.5, 4.0]);
        assert_eq!(a.buf(), &vec![3.5, 6.0, 5.5, 8.0, 7.5, 10.0, 9.5, 12.0]);

        let b = m.sub_row(&[4.0, 5.0]);
        assert_eq!(b.buf(), &vec![-3.0, -3.0, -1.0, -1.0, 1.0, 1.0, 3.0, 3.0]);
        assert_eq!(m.buf(), &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    }

    #[test]
    fn test_vector_scalar_ops() {

        let a = vec![1.0f32, 2.0, 3.0];

        let b = a.mul_scalar(3.0);
        assert_eq!(b, [3.0, 6.0, 9.0]);

        let c = a.add_scalar(3.0);
        assert_eq!(c, [4.0, 5.0, 6.0]);

        let d = a.sub_scalar(3.0);
        assert_eq!(d, [-2.0, -1.0, 0.0]);

        let e = a.div_scalar(2.0);
        assert_eq!(e, [0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_vector_vector_ops() {

        let a = vec![1.5, 2.0, 2.0, 4.0, 5.0];
        let b = vec![3.0, 2.0, 4.0, 5.0, 1.0];

        assert_eq!(a.sub(&b), vec![-1.5, 0.0, -2.0, -1.0, 4.0]);
        assert_eq!(a.add(&b), vec![4.5, 4.0, 6.0, 9.0, 6.0]);
        assert_eq!(a.mul(&b), vec![4.5, 4.0, 8.0, 20.0, 5.0]);
        assert_eq!(b.div(&a), vec![2.0, 1.0, 2.0, 1.25, 0.2]);

        assert_eq!(a.mutate(|x| x * 2.0), vec![3.0, 4.0, 4.0, 8.0, 10.0]);
    }
    
    #[test]
    fn test_vector_ops() {

        let v: Vec<u8> = vec![255, 100, 101, 202];
        let m = v.map(|&x| x as u32);
        assert_eq!(m.sum(), 658);
    }

    #[test]
    fn test_matrix_max() {

        let m = mat![1.0, 2.0, 1.0];
        assert_eq!(m.max().unwrap(), &2.0);
        let n = mat![1.0, f64::NAN, 2.0, 1.0];
        assert_eq!(n.max().unwrap(), &2.0);
        let o = mat![1.0, 2.0, f64::NAN, 1.0];
        assert_eq!(o.max().unwrap(), &2.0);
    }
 
    #[test]
    fn test_matrix_vector_mul() {
        let x = mat![1.0, 2.0, 3.0; 4.0, 2.0, 5.0];
        let h = [2.0, 6.0, 3.0];
        let y = x.mul_vec(&h);
        assert_eq!(y, vec![23.0, 35.0]);
    }

    #[test]
    fn test_transp_matrix_vector_mul() {
        let x = mat![1.0, 2.0; 3.0, 4.0; 2.0, 5.0];
        let h = [2.0, 6.0, 3.0];
        let y = x.transp_mul_vec(&h);
        println!("{:?}", y);
        assert_eq!(y, vec![26.0, 43.0]);
    }

    #[test]
    #[should_panic]
    fn test_matrix_vector_mul_panic() {
        let x = mat![1.0, 2.0, 3.0; 4.0, 2.0, 5.0];
        let i = [2.0, 6.0];
        x.mul_vec(&i);
    }

    #[test]
    fn test_mul_vec_minus_vec() {
        let x = mat![
            1.0, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let v = [2.0, 6.0, 3.0];
        let y = [7.0, 2.0];

        assert_eq!(
            x.mul_vec_minus_vec(&v, &y),
            vec![16.0, 33.0]
        );
    }

    #[test]
    fn test_mul_dgemv() {
        let x = mat![
            1.0, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let v = [2.0, 6.0, 3.0];
        let y = [7.0, 2.0];

        assert_eq!(
            x.mul_dgemv(false, 2.0, &v, -3.0, &y),
            vec![25.0, 64.0]
        );

        let a = [8.0, 3.0];
        let t = [1.0, -4.0, 9.0];
        assert_eq!(
            x.mul_dgemv(true, 2.0, &a, -3.0, &t),
            vec![37.0, 56.0, 51.0]
        );
    }

    #[test]
    fn test_mul_scalar_vec() {
        let x = mat![
            1.0, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let v = [2.0, 6.0, 3.0];

        assert_eq!(
            x.mul_scalar_vec(false, 2.0, &v),
            vec![46.0, 70.0]
        );

        let a = [8.0, 3.0];
        assert_eq!(
            x.mul_scalar_vec(true, 2.0, &a),
            vec![40.0, 44.0, 78.0]
        );
    }

    #[test]
    fn test_sigmoid() {

        assert!(num::abs(1.0.sigmoid() - 0.73106) <= 0.00001);

        assert!(num::abs(1.0.sigmoid_derivative() - 0.196612) <= 0.00002);

        let a = vec![1.0, 2.0, 3.0];
        assert!(a.sigmoid().similar(&vec![0.73106, 0.88080, 0.95257], 0.00001));

        let b = vec![1.0, 2.0];
        assert!(b.sigmoid_derivative().similar(&vec![0.19661, 0.10499], 0.00002));
    }

    #[test]
    fn test_col_mul_row() {

        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 8.0];
        let m = a.col_mul_row(&b);
        assert!(m.eq(&mat![10.0, 16.0; 15.0, 24.0; 20.0, 32.0]));
    }
}

