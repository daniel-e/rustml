//! Module with a collection of different mathematical functions.
//!

extern crate num;

use matrix::Matrix;
use vectors::AddVector;
use self::num::traits::{Num, Float, FromPrimitive};

/// Specifies the dimension over which to perform an operation.
pub enum Dimension {
    Row,
    Column
}

/// Trait to compute the sum of values.
pub trait Sum<T> {
    fn sum(&self, dim: Dimension) -> T;
}

impl <T: Num + Copy> Sum<T> for Vec<T> {

    fn sum(&self, dim: Dimension) -> T {
        match dim {
            Dimension::Row => self.iter().fold(T::zero(), |init, &val| init + val),
            _ => T::zero()
        }
    }
}

// ----------------------------------------------------------------------------

/// Trait to compute the mean of values.
pub trait Mean<T> {
    fn mean(&self, dim: Dimension) -> T;
}

impl <T: Num + Copy + FromPrimitive> Mean<T> for Vec<T> {

    fn mean(&self, dim: Dimension) -> T {
        match dim {
            Dimension::Row => {
                match self.len() {
                    0 => T::zero(),
                    n => self.sum(dim) / (T::from_usize(n).unwrap())
                }
            }
            _ => T::zero()
        }
    }
}

impl <T: Float + FromPrimitive> Mean<Vec<T>> for Matrix<T> {

    fn mean(&self, dim: Dimension) -> Vec<T> {

        if self.rows() == 0 || self.cols() == 0 {
            return vec![];
        }

        match dim {
            Dimension::Column => {
                let mut r: Vec<T> = self.values().take(self.cols()).cloned().collect();
                for row in self.row_iter_at(1) {
                    r.add(row);
                }
                let div = T::from_usize(self.rows()).unwrap();
                for i in r.iter_mut() {
                    *i = *i / div;
                }
                r
            }
            Dimension::Row => {
                vec![T::zero()]
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

    #[test]
    fn test_sum_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(x.sum(Dimension::Row), 10.0);

        let a: Vec<f32> = Vec::new();
        assert_eq!(a.sum(Dimension::Row), 0.0);
        assert_eq!(a.sum(Dimension::Column), 0.0);
    }

    #[test]
    fn test_mean_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(x.mean(Dimension::Row), 2.5);

        let a: Vec<f32> = Vec::new();
        assert_eq!(a.mean(Dimension::Row), 0.0);
        assert_eq!(a.mean(Dimension::Column), 0.0);
    }

    #[test]
    fn test_mean_mat_f32() {

        let a = Matrix::<f32>::new();
        assert_eq!(a.mean(Dimension::Column), vec![]);

        let b = mat![5.0, 6.0];
        assert_eq!(b.mean(Dimension::Column), vec![5.0, 6.0]);

        let x = mat![
            1.0, 2.0;
            3.0, 4.0
        ];
        assert_eq!(x.mean(Dimension::Column), vec![2.0, 3.0]);
    }
}

