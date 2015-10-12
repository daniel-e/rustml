//! Module for linear regression.
//!
extern crate rand;
extern crate libc;

use self::rand::{thread_rng, Rng};
use std::iter::repeat;

use matrix::*;
use ops::{MatrixVectorMul, MatrixVectorOps};

/// Hypothesis for linear regression.
///
/// <script type="text/javascript"
///   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
/// </script>
/// <script type="text/x-mathjax-config">
///   MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
/// </script>
pub struct Hypothesis {
    /// Parameters of the hypothesis.
    thetas: Vec<f64>
}

impl Hypothesis {

    /// Creates a hypothesis with parameters initialized with random values in the
    /// interval [0,1).
    ///
    /// The parameter `n` denotes the number of parameters of the hypothesis.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate rustml;
    /// use rustml::regression::*;
    ///
    /// # fn main() {
    /// let h = Hypothesis::random(5);
    /// assert_eq!(h.params().len(), 5);
    /// assert!(h.params().iter().all(|&x| x < 1.0));
    /// # }
    /// ```
    pub fn random(n: usize) -> Hypothesis {
        Hypothesis {
            thetas: thread_rng().gen_iter::<f64>().take(n).collect()
        }
    }

    /// Creates a new hypothesis with the given parameters.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate rustml;
    /// use rustml::regression::*;
    ///
    /// # fn main() {
    /// let h = Hypothesis::from_params(&[0.1, 0.2, 0.3]);
    /// assert_eq!(h.params().len(), 3);
    /// assert_eq!(h.params(), vec![0.1, 0.2, 0.3]);
    /// # }
    /// ```
    pub fn from_params(params: &[f64]) -> Hypothesis {
        Hypothesis {
            thetas: params.to_vec()
        }
    }

    /// Returns the parameters for this hypothesis.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate rustml;
    /// use rustml::regression::*;
    ///
    /// # fn main() {
    /// let h = Hypothesis::from_params(&[0.1, 0.2, 0.3]);
    /// assert_eq!(h.params(), vec![0.1, 0.2, 0.3]);
    /// # }
    /// ```
    pub fn params(&self) -> Vec<f64> {
        self.thetas.clone()
    }

    /// Computes the hypothesis for the given design matrix using the underlying
    /// BLAS implementation for high performance.
    ///
    /// The following equation is used to compute the hypothesis: 
    /// $h_\theta (X) = X\theta$ where $\theta$ is the row vector of
    /// the parameters.
    /// The result is a vector $y$ where the component $y_i$ is the result
    /// of the hypothesis evaluated for the example at row $i$ of the
    /// matrix, i.e.
    /// $y_i = h_\theta(x\^{(i)}) = \theta\^Tx\^{(i)}$
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    /// use rustml::regression::*;
    ///
    /// # fn main() {
    /// let x = mat![
    ///     1.0, 2.0, 3.0;
    ///     1.0, 5.0, 4.0
    /// ];
    /// let h = Hypothesis::from_params(&[0.1, 0.2, 0.3]);
    /// assert_eq!(
    ///     h.eval(&x), 
    ///     vec![0.1 + 0.4 + 0.9, 0.1 + 1.0 + 1.2]
    /// );
    /// # }
    /// ```

    pub fn eval(&self, x: &Matrix<f64>) -> Vec<f64> {
        x.mul_vec(&self.thetas)
    }

    /// Computes the error of the hypothesis for the examples in the
    /// design matrix `x` and the target values `y`. 
    ///
    /// $$\frac{1}{m}(X\theta-y)\^T(X\theta-y)$$
    pub fn error(&self, x: &Matrix<f64>, y: &[f64]) -> f64 {

        x.mul_vec_minus_vec(&self.thetas, y)
            .iter()
            .fold(0.0, |acc, &x| acc + x*x) / 2.0 / (x.rows() as f64)
    }

    pub fn derivatives(&self, x: &Matrix<f64>, y: &[f64]) -> Vec<f64> {

        let v = x.mul_vec_minus_vec(&self.thetas, &y);
        x.mul_scalar_vec(
            true,
            1.0 / (x.rows() as f64),
            &v
        )
    }
}

/// Trait to create the design matrix of a matrix of features, i.e. a new column is
/// inserted at the left of the matrix where all elements are equal to one.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::*;
/// use rustml::regression::DesignMatrix;
///
/// # fn main() {
/// let m = mat![
///     2.0, 3.0, 4.0;
///     5.0, 6.0, 7.0
/// ];
/// let d = m.design_matrix();
/// assert!(
///     d.eq(
///         &mat![
///             1.0, 2.0, 3.0, 4.0;
///             1.0, 5.0, 6.0, 7.0
///         ]
///     )
/// );
/// # }
/// ```
pub trait DesignMatrix<T> {
    fn design_matrix(&self) -> Self;
}

impl DesignMatrix<f64> for Matrix<f64> {
    fn design_matrix(&self) -> Self {
        self.insert_column(0, 
            &repeat(1.0).take(self.rows()).collect::<Vec<f64>>()
        )
    }
}

impl DesignMatrix<f32> for Matrix<f32> {
    fn design_matrix(&self) -> Self {
        self.insert_column(0, 
            &repeat(1.0).take(self.rows()).collect::<Vec<f32>>()
        )
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

    #[test]
    fn test_hypothesis() {
        let h = Hypothesis::random(5);
        assert!(h.thetas.iter().all(|&x| x < 1.0));

        let i = Hypothesis::from_params(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(i.params(), vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_eval() {
        let x = mat![1.0, 2.0, 3.0; 4.0, 2.0, 5.0];
        let h = Hypothesis::from_params(&[2.0, 6.0, 3.0]);
        let y = h.eval(&x);
        assert_eq!(y, vec![23.0, 35.0]);
    }

    #[test]
    fn test_design_matrix() {
        let x = mat![
            7.0, 2.0, 3.0; 
            4.0, 8.0, 5.0
        ];
        let y = x.design_matrix();
        assert_eq!(
            y.buf(),
            &[1.0, 7.0, 2.0, 3.0, 1.0, 4.0, 8.0, 5.0]
        );
    }

    #[test]
    fn test_error() {
        let x = mat![
            1.0, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let h = Hypothesis::from_params(&[2.0, 6.0, 3.0]);
        let y = [7.0, 2.0];

        assert_eq!(
            h.error(&x, &y),
            336.25
        );
    }

    #[test]
    fn test_derivatives() {
        let x = mat![
            1.0, 2.0, 3.0; 
            4.0, 2.0, 5.0
        ];
        let h = Hypothesis::from_params(&[2.0, 6.0, 3.0]);
        let y = [7.0, 2.0];

        assert_eq!(
            h.derivatives(&x, &y),
            vec![74.0, 49.0, 106.5]
        );
    }
}
