extern crate num;

use math::{Mean, Var};
use math::{Dimension, Normalization};


/// Trait to estimate the mean and the variance of a set of samples.
pub trait Gaussian<T> {
    fn gaussian(&self, nrm: Normalization) -> GaussianParameters<T>;
}

/// Contains all parameters to describe a Gaussian distribution, i.e. the
/// mean and the variance.
///
/// Given a set of samples the parameters can be estimated for all types for
/// which the trait [Gaussian](trait.Gaussian.html) is implemented.
pub struct GaussianParameters<T> {
    mean: T,
    var: T
}

macro_rules! gaussian_impl {
    ($($t:ty)*) => ($(
        impl GaussianParameters<$t> {

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
            pub fn mean(&self) -> $t { self.mean.clone() }

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
            pub fn var(&self) -> $t { self.var.clone() }

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
            pub fn std(&self) -> $t { self.var().sqrt() }

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
            pub fn pr(&self, x: $t) -> $t {

                let s = self.std();
                let m = self.mean();

                let a = s * ((2.0 as $t) * 3.14159265 as $t).sqrt();
                let b = (-(x-m)*(x-m) / ((2.0 as $t) * s * s)).exp();
                b / a
            }
        }

        impl Gaussian<$t> for Vec<$t> {
            fn gaussian(&self, nrm: Normalization) -> GaussianParameters<$t> {
                (&self[..]).gaussian(nrm)
            }
        }

        impl Gaussian<$t> for [$t] {
            fn gaussian(&self, nrm: Normalization) -> GaussianParameters<$t> {
                GaussianParameters {
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
