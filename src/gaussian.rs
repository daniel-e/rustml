//! Module to handle Gaussian distributions.
//!
//! # Examples
//!
//! If the parameters of a Gaussian distribution are already known a Gaussian
//! distribution can be simply created as follows.
//!
//! ```
//! # extern crate rustml;
//! use rustml::gaussian::*;
//!
//! # fn main() {
//! let g = Gaussian::new(0.5 /* mean */, 0.1 /* variance */);
//! assert_eq!(g.mean(), 0.5);
//! assert_eq!(g.var(), 0.1);
//! // standard deviation
//! assert!(g.std() - 0.31623 <= 0.0001);
//! # }
//! ```
//! 

extern crate num;

use self::num::traits::Float;
use math::{Mean, Var};
use math::{Dimension, Normalization};


/// Trait to estimate the mean and the variance of a set of samples.
pub trait GaussianEstimator<T> {
    fn gaussian(&self, nrm: Normalization) -> Gaussian<T>;
}

/// Contains all parameters to describe a Gaussian distribution, i.e. the
/// mean and the variance.
///
/// Given a set of samples the parameters can be estimated for all types for
/// which the trait [Gaussian](trait.Gaussian.html) is implemented.
pub struct Gaussian<T> {
    mean: T,
    var: T
}

impl <T: Float> Gaussian<T> {

    /// Creates a gaussian distribution from the given mean `mean` and the variance
    /// `var`.
    pub fn new(mean: T, var: T) -> Gaussian<T> {
        Gaussian {
            mean: mean,
            var: var
        }
    }
}

pub trait GaussianFunctions<T> {

    /// Returns the mean of the Gaussian distribution.
    /// 
    /// # Examples
    /// ```
    /// # extern crate rustml;
    /// use rustml::gaussian::*;
    /// use rustml::Normalization;
    ///
    /// # fn main() {
    /// let a = vec![1.0f32, 2.0, 4.0, 3.0, 6.0, 5.0];
    /// let g = a.gaussian(Normalization::N);
    /// assert_eq!(g.mean(), 3.5);
    /// # }
    /// ```
    fn mean(&self) -> T;

    /// Returns the variance of the Gaussian distribution.
    ///
    /// # Examples
    /// ```
    /// # extern crate rustml;
    /// use rustml::gaussian::*;
    /// use rustml::Normalization;
    ///
    /// # fn main() {
    /// let a = vec![1.0f32, 2.0, 4.0, 3.0, 6.0, 5.0];
    /// let g = a.gaussian(Normalization::N);
    /// assert!(g.var() - 2.9167 <= 0.0001);
    /// # }
    /// ```
    fn var(&self) -> T;

    /// Returns the standard deviation (the square root of the variance)
    /// of the Gaussian distribution.
    ///
    /// # Examples
    /// ```
    /// # extern crate rustml;
    /// use rustml::gaussian::*;
    /// use rustml::Normalization;
    ///
    /// # fn main() {
    /// let a = vec![1.0f32, 2.0, 4.0, 3.0, 6.0, 5.0];
    /// let g = a.gaussian(Normalization::N);
    /// assert!(g.std() - 1.7078 <= 0.0001);
    /// # }
    /// ```
    fn std(&self) -> T;

    /// Computes the probability density function for the
    /// given value.
    ///
    /// # Examples
    /// ```
    /// # extern crate rustml;
    /// use rustml::gaussian::*;
    /// use rustml::Normalization;
    ///
    /// # fn main() {
    /// let a = vec![1.0f32, 2.0, 4.0, 3.0, 6.0, 5.0];
    /// let g = a.gaussian(Normalization::N);
    /// assert!(g.pr(2.3) - 0.18250 <= 0.00001);
    /// # }
    /// ```
    fn pr(&self, x: T) -> T;
}

macro_rules! gaussian_impl {
    ($($t:ty)*) => ($(
        impl GaussianFunctions<$t> for Gaussian<$t> {

            fn mean(&self) -> $t { self.mean.clone() }

            fn var(&self) -> $t { self.var.clone() }

            fn std(&self) -> $t { self.var().sqrt() }

            fn pr(&self, x: $t) -> $t {

                let s = self.std();
                let m = self.mean();

                let a = s * ((2.0 as $t) * 3.14159265 as $t).sqrt();
                let b = (-(x-m)*(x-m) / ((2.0 as $t) * s * s)).exp();
                b / a
            }
        }

        impl GaussianEstimator<$t> for Vec<$t> {
            fn gaussian(&self, nrm: Normalization) -> Gaussian<$t> {
                (&self[..]).gaussian(nrm)
            }
        }

        impl GaussianEstimator<$t> for [$t] {
            fn gaussian(&self, nrm: Normalization) -> Gaussian<$t> {
                Gaussian {
                    mean: self.mean(Dimension::Row),
                    var: self.var(Dimension::Row, nrm)
                }
            }
        }
    )*)
}

gaussian_impl!{ f32 f64 }

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use math::Normalization;

    #[test]
    fn test_parameters() {
        let a = vec![1.0f32, 2.0, 3.0, 2.0, 1.0, 2.0];
        let mut p = a.gaussian(Normalization::N);
        assert!(p.mean() - 1.8333 <= 0.0001);
        assert!(p.var() - 0.47222 <= 0.00001);
        assert!(p.std() - 0.68718 <= 0.00001);
        assert!(p.pr(2.0) - 0.56372 <= 0.00001);
        assert!(p.pr(1.5) - 0.51613 <= 0.00001);

        p = a.gaussian(Normalization::MinusOne);
        assert!(p.mean() - 1.8333 <= 0.0001);
        assert!(p.var() - 0.56667 <= 0.00001);
        assert!(p.std() - 0.75277 <= 0.00001);
        assert!(p.pr(2.0) - 0.51713 <= 0.00001);
        assert!(p.pr(1.5) - 0.48048 <= 0.00001);
    }
}
