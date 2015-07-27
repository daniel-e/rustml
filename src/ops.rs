extern crate num;

use matrix::Matrix;
use ops_inplace::VectorVectorOpsInPlace;

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

/// Trait for operations on matices.
pub trait MatrixOps<T> {

    /// Computes the reciprocal (inverse) of each element of the matrix
    /// and returns the result.
    fn recip(&self) -> Matrix<T>;
}

macro_rules! matrix_ops_impl {
    ($($t:ty)*) => ($(

        impl MatrixOps<$t> for Matrix<$t> {
            fn recip(&self) -> Matrix<$t> {
                self.map(|&x| (1.0 as $t) / x)
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

    fn mul(&self, rhs: &[T]) -> Vec<T>;

    fn div(&self, rhs: &[T]) -> Vec<T>;

    fn mutate<F>(&self, f: F) -> Vec<T>
        where F: Fn(T) -> T;
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
        }

        impl VectorVectorOps<$t> for Vec<$t> {
            fn sub(&self, v: &[$t])                 -> Vec<$t> { (self[..]).sub(v)    }
            fn add(&self, v: &[$t])                 -> Vec<$t> { (self[..]).add(v)    }
            fn mul(&self, v: &[$t])                 -> Vec<$t> { (self[..]).mul(v)    }
            fn div(&self, v: &[$t])                 -> Vec<$t> { (self[..]).div(v)    }
            fn mutate<F: Fn($t) -> $t>(&self, f: F) -> Vec<$t> { (self[..]).mutate(f) }
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
}

macro_rules! matrix_vector_ops_impl {
    ($($t:ty)*) => ($(

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
        }
    )*)
}

matrix_vector_ops_impl!{ f32 f64 }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::Matrix;
    use math::*;

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
}

