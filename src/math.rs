//! Module with a collection of different mathematical functions.
//!

extern crate num;

use matrix::Matrix;
use vectors::AddVector;
use self::num::traits::{Num, Float, FromPrimitive};

/// Determines the dimension over which to perform an operation.
pub enum Dimension {
    /// Perform the operation over all elements of a row.
    Row,
    /// Perform the operatino over all elements of a column.
    Column
}

/// Determines the type of normalization used for computing the variance
/// or standard deviation.
pub enum Normalization {
    /// Use as denominator n, i.e. the number of examples.
    N,
    /// Use as denominator (n-1), i.e. the number of examples minus one.
    MinusOne
}

// ----------------------------------------------------------------------------

/// Trait to compute the sum of values.
pub trait Sum<T> {
    /// Computes the sum over all elements for the specified dimension.
    ///
    /// When computing the sum of a vector or a slice both
    /// are interpreted as a row vector. Thus,
    /// the only valid dimension for vectors or slices
    /// is `Dimension::Row`. For other dimensions the function 
    /// will always return zero.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let v = vec![1.0, 5.0, 9.0];
    /// assert_eq!(v.sum(Dimension::Row), 15.0);
    /// assert_eq!(v.sum(Dimension::Column), 0.0);
    /// ```
    fn sum(&self, dim: Dimension) -> T;
}

impl <T: Num + Copy> Sum<T> for Vec<T> {

    fn sum(&self, dim: Dimension) -> T {
        (&self[..]).sum(dim)
    }
}

impl <T: Num + Copy> Sum<T> for [T] {

    fn sum(&self, dim: Dimension) -> T {
        match dim {
            // TODO is there a more efficient method?
            Dimension::Row => self.iter().fold(T::zero(), |init, &val| init + val),
            _ => T::zero()
        }
    }
}

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
    /// let v = vec![1.0, 5.0, 9.0];
    /// assert_eq!(v.sum(Dimension::Row), 15.0);
    /// assert_eq!(v.sum(Dimension::Column), 0.0);
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

impl <T: Float + FromPrimitive> Mean<Vec<T>> for Matrix<T> {

    fn mean(&self, dim: Dimension) -> Vec<T> {

        if self.rows() == 0 || self.cols() == 0 {
            return vec![];
        }

        match dim {
            Dimension::Column => {
                // TODO reimplement when Sum for Matrix is implemented
                let mut r: Vec<T> = self.values().take(self.cols()).cloned().collect();
                for row in self.row_iter_at(1) {
                    r.add(row);
                }
                let n = T::from_usize(self.rows()).unwrap();
                for i in r.iter_mut() {
                    *i = *i / n;
                }
                r
            }
            Dimension::Row => {
                // TODO reimplement when Sum for Matrix is implemented
                let n = T::from_usize(self.cols()).unwrap();
                self.row_iter().map(|row| row.sum(Dimension::Row) / n).collect()
            }
        }
    }
}

// ----------------------------------------------------------------------------

/// Trait to compute sigma squared of values.
pub trait Var<T> {
    fn var(&self, dim: Dimension, nrm: Normalization) -> T;
}

impl <T: Num + Copy + FromPrimitive> Var<T> for Vec<T> {

    fn var(&self, dim: Dimension, nrm: Normalization) -> T {
        match dim {
            Dimension::Row => {
                match self.len() {
                    0 => T::zero(),
                    n => {
                        let mu = self.mean(dim);
                        let mut d = T::from_usize(n).unwrap();
                        match nrm {
                            Normalization::MinusOne => {
                                if n > 1 {
                                    d = d - T::one();
                                }
                            }
                            _ => ()
                        }
                        self.iter().map(|&x| (x - mu) * (x - mu)).fold(T::zero(), |acc, x| acc + x) / d
                    }
                }
            }
            _ => T::zero()
        }
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

    #[test]
    fn test_var_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(x.var(Dimension::Row, Normalization::N), 1.25);

        let y: Vec<f32> = vec![10.0, 5.0, 3.0, 11.0, 2.0, 15.0, 13.0, 5.0, 3.0];
        assert!(y.var(Dimension::Row, Normalization::N) - 20.914 < 0.001);

        let a: Vec<f32> = Vec::new();
        assert_eq!(a.var(Dimension::Row, Normalization::N), 0.0);
        assert_eq!(a.var(Dimension::Column, Normalization::N), 0.0);
        assert_eq!(x.var(Dimension::Column, Normalization::N), 0.0);
    }

    #[test]
    fn test_var_minus_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert!(x.var(Dimension::Row, Normalization::MinusOne) - 1.6667 < 0.0001);

        let y: Vec<f32> = vec![10.0, 5.0, 3.0, 11.0, 2.0, 15.0, 13.0, 5.0, 3.0];
        assert!(y.var(Dimension::Row, Normalization::MinusOne) - 23.528 < 0.001);

        let a: Vec<f32> = Vec::new();
        assert_eq!(a.var(Dimension::Row, Normalization::MinusOne), 0.0);
        assert_eq!(a.var(Dimension::Column, Normalization::MinusOne), 0.0);
        assert_eq!(x.var(Dimension::Column, Normalization::MinusOne), 0.0);
    }

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

