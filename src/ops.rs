extern crate num;

use matrix::Matrix;

// ----------------------------------------------------------------------------

/// Trait for matrix scalar operations.
pub trait MatrixScalarOps<T> {
    /// Adds a scalar to each element of the matrix and returns
    /// the result.
    fn add_scalar(&self, scalar: T) -> Matrix<T>;

    /// Multiplies each element of the matrix with a scalar
    /// and returns the result.
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

/// Trait for vector scalar operations.
pub trait VectorScalarOps<T> {
    /// Adds a scalar to each element of the vector and returns
    /// the result.
    fn mul_scalar(&self, scalar: T) -> Vec<T>;

    /// Multiplies each element of the vector with a scalar
    /// and returns the result.
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

/// Trait for vector vector operations.
pub trait VectorVectorOps<T> {

    fn sub(&self, rhs: &[T]) -> Vec<T>;

    fn mutate<F>(&self, f: F) -> Vec<T>
        where F: Fn(T) -> T;
}

macro_rules! vector_vector_ops_impl {
    ($($t:ty)*) => ($(

        impl VectorVectorOps<$t> for [$t] {
            fn sub(&self, v: &[$t]) -> Vec<$t> {
                self.iter().zip(v.iter()).map(|(&x, &y)| x - y).collect()
            }

            fn mutate<F>(&self, f: F) -> Vec<$t>
                where F: Fn($t) -> $t {

                self.iter().map(|&x| f(x)).collect()
            }
        }

        impl VectorVectorOps<$t> for Vec<$t> {
            fn sub(&self, v: &[$t])                 -> Vec<$t> { (self[..]).sub(v)    }
            fn mutate<F: Fn($t) -> $t>(&self, f: F) -> Vec<$t> { (self[..]).mutate(f) }
        }
    )*)
}

vector_vector_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

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
    fn test_vector_vector_ops() {

        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 2, 4, 5, 1];
        assert_eq!(a.sub(&b), vec![-2, 0, -1, -1, 4]);

        assert_eq!(a.mutate(|x| x * 2), vec![2, 4, 6, 8, 10]);
    }
}

