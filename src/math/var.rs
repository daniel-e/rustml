extern crate num;

use math::{Dimension, Normalization, Mean};
use vectors::zero;
use matrix::Matrix;
use ops::{VectorVectorOps, VectorScalarOps};
use ops_inplace::VectorVectorOpsInPlace;

// ----------------------------------------------------------------------------

/// Trait to compute the variance of values.
pub trait Var<T> {
    /// <script type="text/javascript"
    ///   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    /// </script>
    /// <script type="text/x-mathjax-config">
    ///   MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
    /// </script>
    ///
    /// If the implementation for a vector is used this method computes the
    /// variance of the values of that vectors. For vectors only the
    /// dimension Dimension::Row is valid.
    ///
    /// If the implementation for a matrix is used this methods
    /// computes the variance of all values of each row if the dimension is
    /// equal to `Dimension::Row` or each column if the dimension is
    /// `Dimension::Column`.
    ///
    /// For the parameter `Normalization::N` the following formula is used to compute the
    /// variance:
    /// $$\frac{1}{n} \sum_{i=1}\^n \(x_i - \mu\)\^2$$
    ///
    /// For the parameter `Normalization::MinusOne` the following formula is used to compute
    /// the variance:
    /// $$\frac{1}{n-1} \sum_{i=1}\^n \(x_i - \mu\)\^2$$
    /// If n is equal to 1 the parameter `Normalization::MinusOne` behaves like `Normalization::N`.
    ///
    /// # Compute the variance of the values in a vector.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// # extern crate num;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let a = vec![1.0, 2.0, 1.5, 1.8, 2.2];
    ///
    /// let mut r = a.var(Dimension::Row, Normalization::N);
    /// assert!(num::abs(r - 0.176) < 0.001);
    ///
    /// r = a.var(Dimension::Row, Normalization::MinusOne);
    /// assert!(num::abs(r - 0.22) < 0.001);
    /// # }
    /// ```
    ///
    /// # Compute the variance for rows and columns of a matrix.
    /// <br>
    /// Let M be a matrix with the rows $r_1, r_2, \dots$, i.e.
    ///
    /// $ M = 
    /// \begin{bmatrix} -\  r_1 - \\\\ 
    ///    -\  r_2 - \\\\ 
    ///    \vdots \\\\
    /// \end{bmatrix}
    /// $
    ///
    /// Then, the variance of the matrix with the dimension parameter `Dimension::Row` and
    /// the normalization parameter `x` returns the following vector: vec![r1.var(Row, x), r2.var(Row, x), ...]
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// # extern crate num;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0f32, 5.0, 3.0, 6.0, 2.0;
    ///     2.0, 2.5, 2.2, 1.8, 2.1;
    ///     4.0, 9.0, 2.7, 13.0, 1.9
    /// ];
    ///
    /// let mut r = m.var(Dimension::Row, Normalization::N);
    /// let mut e = vec![3.44, 0.05, 17.93];
    /// // check that r is "equal" to e
    /// assert!(r.iter().zip(e.iter()).all(|(&a, &b)| num::abs(a - b) < 0.01));
    ///
    /// r = m.var(Dimension::Column, Normalization::N);
    /// e = vec![1.56, 7.17, 0.11, 21.34, 0.007];
    /// // check that r is "equal" to e
    /// assert!(r.iter().zip(e.iter()).all(|(&a, &b)| num::abs(a - b) < 0.01));
    /// # }
    /// ```
    fn var(&self, dim: Dimension, nrm: Normalization) -> T;
}

macro_rules! var_impl {
    ($($t:ty)*) => ($(

        impl Var<$t> for Vec<$t> {

            fn var(&self, dim: Dimension, nrm: Normalization) -> $t {
                (&self[..]).var(dim, nrm)
            }
        }

        impl Var<$t> for [$t] {

            fn var(&self, dim: Dimension, nrm: Normalization) -> $t {

                if self.len() == 0 {
                    return 0 as $t;
                }

                match dim {
                    Dimension::Row => {
                        let m = self.mean(dim);
                        let n = self.len();
                        let d = match nrm {
                            Normalization::N        => n as $t,
                            Normalization::MinusOne => if n > 1 { (n - 1) as $t } else { n as $t }
                        };
                        self.iter()
                            .map(|&x| (x - m) * (x - m))
                            .fold(0 as $t, |acc, x| acc + x) / d
                    }
                    _ => 0 as $t
                }
            }
        }

        impl Var<Vec<$t>> for Matrix<$t> {

            fn var(&self, dim: Dimension, nrm: Normalization) -> Vec<$t> {

                match dim {
                    Dimension::Row => {
                        self.row_iter().map(|row| row.var(Dimension::Row, nrm)).collect()
                    }
                    Dimension::Column => {
                        let mean_vec = self.mean(Dimension::Column);
                        let mut v = zero::<$t>(self.cols());
                        for row in self.row_iter() {
                            let x = row.sub(&mean_vec).mutate(|x| x * x);
                            v.add(&x);
                        }
                        let n = self.rows();
                        let d = match nrm {
                            Normalization::N        => n as $t,
                            Normalization::MinusOne => if n > 1 { (n - 1) as $t } else { n as $t }
                        };
                        v.div_scalar(d)
                    }
                }
            }
        }
    )*)
}

var_impl!{ f32 f64 }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate num;

    use super::*;
    use math::{Dimension, Normalization};
    use matrix::Matrix;

    fn vec_equal(a: &Vec<f32>, b: &Vec<f32>) -> bool {

        a.iter().zip(b.iter()).all(|(&x, &y)| num::abs(x - y) < 0.01)
    }

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
    fn test_var_matrix() {

        let m = mat![
            1.0f32, 5.0, 3.0, 6.0, 2.0;
            2.0, 2.5, 2.2, 1.8, 2.1;
            4.0, 9.0, 2.7, 13.0, 1.9
        ];

        let mut x = m.var(Dimension::Row, Normalization::N);
        assert_eq!(x.len(), 3);
        assert!(vec_equal(&x, &vec![3.44, 0.0536, 17.926]));
        let mut y = m.var(Dimension::Row, Normalization::MinusOne);
        assert_eq!(y.len(), 3);
        assert!(vec_equal(&y, &vec![4.3, 0.067, 22.407]));

        x = m.var(Dimension::Column, Normalization::N);
        assert_eq!(x.len(), 5);
    assert_eq!(x.to_vec(), vec![1.5556, 7.1667, 0.10889, 21.342, 0.0066667]);
        assert!(vec_equal(&x, &vec![1.5556, 7.1667, 0.10889, 21.342, 0.0066667]));
        y = m.var(Dimension::Column, Normalization::MinusOne);
        assert_eq!(y.len(), 5);
        assert!(vec_equal(&y, &vec![2.3333, 10.750, 0.16333, 32.013, 0.01]));
    }
}

