extern crate num;

use matrix::Matrix;


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

// TODO div_scalar
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

//    fn add(&self, rhs: &[T]) -> Vec<T>;

    fn mutate<F>(&self, f: F) -> Vec<T>
        where F: Fn(T) -> T;
}

macro_rules! vector_vector_ops_impl {
    ($($t:ty)*) => ($(

        impl VectorVectorOps<$t> for [$t] {
            fn sub(&self, v: &[$t]) -> Vec<$t> {
                self.iter().zip(v.iter()).map(|(&x, &y)| x - y).collect()
            }

//            fn add(&self, v: &[$t]) -> Vec<$t> {
//                self.iter().zip(v.iter()).map(|(&x, &y)| x + y).collect()
//            }

            fn mutate<F>(&self, f: F) -> Vec<$t>
                where F: Fn($t) -> $t {

                self.iter().map(|&x| f(x)).collect()
            }
        }

        impl VectorVectorOps<$t> for Vec<$t> {
            fn sub(&self, v: &[$t])                 -> Vec<$t> { (self[..]).sub(v)    }
//            fn add(&self, v: &[$t])                 -> Vec<$t> { (self[..]).add(v)    }
            fn mutate<F: Fn($t) -> $t>(&self, f: F) -> Vec<$t> { (self[..]).mutate(f) }
        }
    )*)
}

vector_vector_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

// ----------------------------------------------------------------------------

pub trait MatrixVectorOps<T> {

    /// Adds the given vector to each row of the matrix.
    fn add_row(&self, rhs: &[T]) -> Matrix<T>;
}

macro_rules! matrix_vector_ops_impl {
    ($($t:ty)*) => ($(

        impl MatrixVectorOps<$t> for Matrix<$t> {

            fn add_row(&self, rhs: &[$t]) -> Matrix<$t> {

                let mut m = self.clone();
                //XXX
                //mut_row_iter()
                m
            }
        }
    )*)
}

matrix_vector_ops_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

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
        let c = m.sub_scalar(3.0);
        assert_eq!(c.buf(), &vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_vector_ops() {

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

        let a = vec![1.5, 2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 2.0, 4.0, 5.0, 1.0];

        assert_eq!(a.sub(&b), vec![-1.5, 0.0, -1.0, -1.0, 4.0]);
//        assert_eq!(a.add(&b), vec![4.5, 4.0, 7.0, 9.0, 6.0]);

        assert_eq!(a.mutate(|x| x * 2.0), vec![3.0, 4.0, 6.0, 8.0, 10.0]);
    }
}

