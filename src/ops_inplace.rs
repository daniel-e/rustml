extern crate libc;

use self::libc::{c_int, c_float, c_double};

use blas::{cblas_saxpy, cblas_daxpy};

/// Trait to add a slice to a vector using the underlying BLAS implementation.
pub trait VectorVectorOpsInPlace<T> {

    /// Adds the given slice `rhs` inplace.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::ops_inplace::VectorVectorOpsInPlace;
    ///
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.iadd(&y);
    /// assert_eq!(v, vec![4.0, 10.0]);
    /// ```
    fn iadd(&mut self, rhs: &[T]);

    /// Substracts the given slice `rhs` inplace.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::ops_inplace::VectorVectorOpsInPlace;
    ///
    /// let mut v = vec![1.0, 2.0];
    /// let y = vec![3.0, 8.0];
    /// v.isub(&y);
    /// assert_eq!(v, vec![-2.0, -6.0]);
    /// ```
    fn isub(&mut self, rhs: &[T]);

    fn imul(&mut self, rhs: &[T]);
    fn idiv(&mut self, rhs: &[T]);
}

macro_rules! impl_vector_vector_ops_inplace {
    ( $( $x:ty, $y:expr, $z:ty )+ ) => ($(

        impl VectorVectorOpsInPlace<$x> for Vec<$x> {

            fn iadd(&mut self, rhs: &[$x]) {
                (self[..]).iadd(rhs);
            }

            fn isub(&mut self, rhs: &[$x]) {
                (self[..]).isub(rhs);
            }

            fn imul(&mut self, rhs: &[$x]) {
                (self[..]).imul(rhs);
            }

            fn idiv(&mut self, rhs: &[$x]) {
                (self[..]).idiv(rhs);
            }
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
}

