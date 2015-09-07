use math::{Mean, Var};
use math::{Dimension, Normalization};


/// Trait to estimate the mean and the variance of a set of samples.
pub trait Scale<T> {
    fn scale(&self, nrm: Normalization) -> T;
}

impl Scale<Matrix<f32>> for Matrix<f32> {

    fn scale(&self, nrm: Normalization) -> Matrix<f32> {

        let mean_vec = self.mean(Dimension::Column);

    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use math::Normalization;

    #[test]
    fn test_parameters() {
    }
}
