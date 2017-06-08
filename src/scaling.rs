//! Module to scale vectors and matrices.

extern crate num;

use self::num::traits::Float;
use gaussian::{Gaussian, GaussianFunctions};
use matrix::Matrix;
use math::{Dimension, Normalization, Mean, Var};
use ops_inplace::VectorVectorOpsInPlace;

/// Trait to scale a matrix.
pub trait ScaleMatrix<T> {

    /// Scales the values of the matrix and returns the scaled
    /// matrix and the parameters that were used for scaling.
    ///
    /// First the mean and the standard deviation of the values for
    /// each column is computed. Then, a new matrix is computed where 
    /// the element at row `i` and column `j` is the result of
    /// `(m(i,j) - mu(j)) / std(j)` where 
    ///
    /// * m(i,j) is the element at row `i` and column `j` in the original matrix `m`
    /// * mu(j) is the mean of the elements in column `j` of `m`
    /// * std(j) is the standard deviation of the elements in column `j` of `m`
    ///
    /// If the standard deviation is smaller than 0.0001 then 1.0 is used
    /// in the denominator.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    /// use rustml::scaling::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 100.0;
    ///     2.0, 150.0;
    ///     0.6, 110.0
    /// ];
    ///
    /// let (scaled, gaussians) = m.scale();
    ///
    /// // mean and standard deviation of first column
    /// assert_eq!(gaussians[0].mean(), 1.2);
    /// assert!(gaussians[0].std() - 0.721 < 0.001);
    /// // mean and standard deviation of second column
    /// assert_eq!(gaussians[1].mean(), 120.0);
    /// assert!(gaussians[1].std() - 26.458 < 0.001);
    ///
    /// let e = mat![
    ///     -0.27735, -0.75593;
    ///     1.10940, 1.13389;
    ///     -0.83205, -0.37796
    /// ];
    ///
    /// assert!(scaled.similar(&e, 0.0001))
    /// # }
    /// ```
    fn scale(&self) -> (Box<Self>, Vec<Gaussian<T>>);
}

macro_rules! scale_mat_impl {
    ($($t:ty)*) => ($(

        impl ScaleMatrix<$t> for Matrix<$t> {

            // TODO normalization, dimension
            fn scale(&self) -> (Box<Matrix<$t>>, Vec<Gaussian<$t>>) {

                let mean_vec = self.mean(Dimension::Column);
                let var_vec = self.var(Dimension::Column, Normalization::MinusOne);
                let std_vec: Vec<$t> = var_vec.iter().
                    map(|&x| x.sqrt()).map(|x| if num::abs(x) < 0.0001 { 1 as $t } else { x }).collect();
                // TODO 0.000001 is hard coded

                let r = mean_vec.iter().zip(var_vec.iter()).map(|(&x, &y)| Gaussian::new(x, y)).collect();

                let mut mr = self.clone();
                for i in 0..mr.rows() {
                    let r = mr.row_mut(i).unwrap();
                    r.isub(&mean_vec);
                    r.idiv(&std_vec);
                }
                (Box::new(mr), r)
            }
        }

    )*)
}

scale_mat_impl!{ f32 f64 }

// ----------------------------------------------------------------------------

/// Trait to scale a vector.
pub trait ScaleVector<T> {
    /// Scales the values of the vector by the parameters provided
    /// by the vector `g`.
    ///
    /// For each value the following is done: the value at index `i` is
    /// subtracted by the mean
    /// of the `Gaussian` at index `i` and the result is divided by the
    /// standard deviation of the `Gaussian` at index `i`. This is done
    /// for each value in the vector and a new vector is returned with
    /// the result.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::scaling::*;
    /// use rustml::gaussian::*;
    ///
    /// # fn main() {
    /// let gaussians = vec![
    ///     Gaussian::new(2.0, 4.0),
    ///     Gaussian::new(5.0, 16.0)
    /// ];
    ///
    /// assert_eq!(
    ///     [4.0, 7.0].scale_by(&gaussians), 
    ///     vec![
    ///         (4.0 - 2.0) / 2.0,
    ///         (7.0 - 5.0) / 4.0
    ///     ]
    /// );
    /// # }
    /// ```
    fn scale_by(&self, g: &Vec<Gaussian<T>>) -> Vec<T>;
}

macro_rules! scaling_vec_impl {
    ($($t:ty)*) => ($(
        impl ScaleVector<$t> for Vec<$t> {
            fn scale_by(&self, g: &Vec<Gaussian<$t>>) -> Vec<$t> {
                (self[..]).scale_by(g)
            }
        }

        impl ScaleVector<$t> for [$t] {
            fn scale_by(&self, g: &Vec<Gaussian<$t>>) -> Vec<$t> {
                self.iter().zip(g.iter()).map(|(&x, ref y)| (x - y.mean()) / y.std()).collect()
            }
        }
    )*)
}

scaling_vec_impl!{ f32 f64 }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate num;

    use super::*;
    use matrix::Matrix;
    use gaussian::{Gaussian, GaussianFunctions};

    #[test]
    fn test_scale_mat() {
        let m = mat![
            1.0, 2.0; 
            3.0, 5.0;
            4.0, 10.0
        ];
        let (x, g) = m.scale();
        assert!(g[0].mean() - 2.6667 < 0.001);
        assert!(g[1].mean() - 5.6667 < 0.001);
        assert!(g[0].var() - 2.3333 < 0.001);
        assert!(g[1].var() - 16.3333 < 0.001);
        assert!(g[0].std() - 1.5275 < 0.001);
        assert!(g[1].std() - 4.0415 < 0.001);
        
        assert!(x.get(0, 0).unwrap() - (-1.09109) < 0.001);
        assert!(x.get(0, 1).unwrap() - (-0.90726) < 0.001);
        assert!(x.get(1, 0).unwrap() - (0.21822) < 0.001);
        assert!(x.get(1, 1).unwrap() - (-0.16496) < 0.001);
        assert!(x.get(2, 0).unwrap() - (0.87287) < 0.001);
        assert!(x.get(2, 1).unwrap() - (1.07222) < 0.001);
    }

    #[test]
    fn test_scale_mat2() {
        let m = mat![
            1.0, 50.0, 1000.0;
            0.3, 45.0, 1200.0;
            1.9, 44.0, 1150.0;
            0.7, 60.0, 1300.0
        ];
        let (s, _g) = m.scale();
        let r = mat![
            0.036761, 0.034153, -1.300000;
            -0.992540, -0.648901, 0.300000;
            1.360147, -0.785512, -0.100000;
            -0.404368, 1.400261, 1.100000
        ];
        assert!(s.buf().iter().zip(r.buf().iter()).all(|(&a, &b)| num::abs(a - b) < 0.01));
    }

    #[test]
    fn test_scale_vec() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let g = vec![
            Gaussian::new(2.0, 16.0),
            Gaussian::new(3.0, 4.0),
            Gaussian::new(9.0, 64.0),
            Gaussian::new(5.0, 25.0)
        ];
        let y = x.scale_by(&g);
        assert_eq!(y, vec![-0.25, -0.5, -0.75, -0.2]);
    }

}

 
