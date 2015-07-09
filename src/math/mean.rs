extern crate num;

use self::num::traits::{Num, FromPrimitive};

use matrix::Matrix;
use vectors::zero;
use ops::VectorVectorOps;
use math::Dimension;
use math::Sum;

// ----------------------------------------------------------------------------

/// Trait to compute the mean of values.
pub trait Mean<T> {
    /// Computes the mean over all elements for the specified dimension.
    ///
    /// When computing the mean of a vector or a slice both
    /// are interpreted as a row vector. Thus,
    /// the only valid dimension for vectors or slices
    /// is `Dimension::Row`. For other dimensions or if the vector or
    /// slice is empty the function will always return zero. 
    ///
    /// # Examples
    ///
    /// ```
    /// use rustml::math::{Mean, Dimension};
    ///
    /// let v = vec![1.0, 5.0, 9.0];
    /// assert_eq!(v.mean(Dimension::Row), 5.0);
    /// assert_eq!(v.mean(Dimension::Column), 0.0);
    /// ```
    ///
    /// # Implementation details
    ///
    /// Uses the [Sum](trait.Sum.html) trait to compute the sum which is then
    /// divided by the number of values.
    fn mean(&self, dim: Dimension) -> T;
    // TODO document matrix
}

impl <T: Num + Copy + FromPrimitive> Mean<T> for Vec<T> {

    fn mean(&self, dim: Dimension) -> T {
        (&self[..]).mean(dim)
    }
}

impl <T: Num + Copy + FromPrimitive> Mean<T> for [T] {

    fn mean(&self, dim: Dimension) -> T {
        match dim {
            Dimension::Row => {
                match self.len() {
                    0 => T::zero(),
                    // TODO unwrap
                    n => self.sum(dim) / (T::from_usize(n).unwrap())
                }
            }
            _ => T::zero()
        }
    }
}

macro_rules! mean_impl {
    ($($t:ty)*) => ($(
        impl Mean<Vec<$t>> for Matrix<$t> {

            fn mean(&self, dim: Dimension) -> Vec<$t> {

                if self.rows() == 0 || self.cols() == 0 {
                    return vec![];
                }

                match dim {
                    Dimension::Column => {
                        // TODO reimplement when Sum for Matrix is implemented
                        let mut r: Vec<$t> = self.values().take(self.cols()).cloned().collect();
                        for row in self.row_iter_at(1) {
                            r.add(row);
                        }
                        let n = self.rows() as $t;
                        for i in r.iter_mut() {
                            *i = *i / n;
                        }
                        r
                    }
                    Dimension::Row => {
                        // TODO reimplement when Sum for Matrix is implemented
                        let n = self.cols() as $t;
                        self.row_iter().map(|row| row.sum(Dimension::Row) / n).collect()
                    }
                }
            }
        }
    )*)
}

mean_impl!{ f32 f64 }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;
    use math::Dimension;

    #[test]
    fn test_mean_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(x.mean(Dimension::Row), 2.5);

        let a: Vec<f32> = Vec::new();
        assert_eq!(a.mean(Dimension::Row), 0.0);
        assert_eq!(a.mean(Dimension::Column), 0.0);

        let b = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(b.mean(Dimension::Row), 2.5);
    }

    #[test]
    fn test_mean_mat_f32() {

        let a = Matrix::<f32>::new();
        assert_eq!(a.mean(Dimension::Column), vec![]);
        assert_eq!(a.mean(Dimension::Row), vec![]);

        let b = mat![5.0, 6.0];
        assert_eq!(b.mean(Dimension::Column), vec![5.0, 6.0]);
        assert_eq!(b.mean(Dimension::Row), vec![5.5]);

        let x = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        assert_eq!(x.mean(Dimension::Column), vec![2.0, 3.0]);
        assert_eq!(x.mean(Dimension::Row), vec![1.5, 3.5]);
    }
}

