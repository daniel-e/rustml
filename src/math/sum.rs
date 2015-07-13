extern crate num;

use matrix::Matrix;
use vectors::zero;
use ops::VectorVectorOps;
use math::Dimension;

/// Trait to compute the sum of values.
pub trait Sum<T> {
    /// Computes the sum over all elements of the specified dimension.
    ///
    /// When computing the sum of a vector or a slice both
    /// are interpreted as a row vector. Thus,
    /// the only valid dimension for vectors or slices
    /// is `Dimension::Row`. For other dimensions the function 
    /// will always return zero.
    ///
    /// # Examples
    ///
    /// Compute the sum of all elements of a vector.
    ///
    /// ```
    /// use rustml::*;
    ///
    /// let v = vec![1.0, 5.0, 9.0];
    /// assert_eq!(v.sum(Dimension::Row), 15.0);
    /// // The dimension `Column` does not exist for a vector. The result
    /// // is always zero.
    /// assert_eq!(v.sum(Dimension::Column), 0.0);
    /// ```
    ///
    /// To compute the sum of all elements in each column or each row of a
    /// matrix you have to specify the correct dimension. 
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 2.0; 
    ///     5.0, 10.0
    /// ];
    ///
    /// assert_eq!(m.sum(Dimension::Row), vec![3.0, 15.0]);
    /// assert_eq!(m.sum(Dimension::Column), vec![6.0, 12.0]);
    /// # }
    /// ```
    ///
    /// # Implementation details
    ///
    /// Currently, the following code is used to compute the sum of a row:
    /// 
    /// ```ignore
    /// self.iter().fold(T::zero(), |init, &val| init + val)
    /// ```
    ///
    /// In the future this may be replaced by a more efficient implementation.
    fn sum(&self, dim: Dimension) -> T;
}

macro_rules! sum_vec_impl {
    ($($t:ty)*) => ($(
        impl Sum<$t> for Vec<$t> {

            // TODO in the future there might be a default value
            fn sum(&self, dim: Dimension) -> $t {
                (&self[..]).sum(dim)
            }
        }

        impl Sum<$t> for [$t] {

            // TODO in the future there might be a default value
            fn sum(&self, dim: Dimension) -> $t {
                match dim {
                    // TODO is there a more efficient method to sum up all values?
                    Dimension::Row => {
                        self.iter().fold(0 as $t, |init, &val| init + val)
                    }
                    _ => 0 as $t
                }
            }
        }
    )*)
}

sum_vec_impl!{ usize u8 u16 u32 u64 isize i8 i16 i32 i64 f32 f64 }

macro_rules! sum_impl {
    ($($t:ty)*) => ($(
        impl Sum<Vec<$t>> for Matrix<$t> {

            fn sum(&self, dim: Dimension) -> Vec<$t> {
                match dim {
                    Dimension:: Column => {
                        let mut v = zero::<$t>(self.cols());
                        for row in self.row_iter() {
                            v.add(row);
                        }
                        v
                    }
                    Dimension::Row => {
                        self.row_iter().map(|row| row.sum(Dimension::Row)).collect()
                    }
                }
            }
        }
    )*)
}

sum_impl!{ f32 f64 }

// ------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;
    use math::Dimension;

    #[test]
    fn test_sum_matrix_f32() {

        let m = mat![
            1.0f32, 2.0; 
            5.0, 10.0
        ];
        assert_eq!(m.sum(Dimension::Column), vec![6.0, 12.0]);
        assert_eq!(m.sum(Dimension::Row), vec![3.0, 15.0]);

        let a = Matrix::<f32>::new();
        assert_eq!(a.sum(Dimension::Column), vec![]);
        assert_eq!(a.sum(Dimension::Row), vec![]);
    }

    #[test]
    fn test_sum_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(x.sum(Dimension::Row), 10.0);

        let a: Vec<f32> = Vec::new();
        assert_eq!(a.sum(Dimension::Row), 0.0);
        assert_eq!(a.sum(Dimension::Column), 0.0);
    }
}

