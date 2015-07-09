extern crate num;

use self::num::traits::{Num, FromPrimitive};
use math::{Mean, Var};
use math::{Dimension, Normalization};


pub struct Gaussian<T> {
    mean: T,
    var: T
}


impl <T: Num + Copy + FromPrimitive> Gaussian<T> {

    pub fn estimate(x: &[T], nrm: Normalization) -> Gaussian<T> {

        Gaussian {
            mean: x.mean(Dimension::Row),
            var: x.var(Dimension::Row, nrm)
        }
    }

    pub fn pr(&self, x: &T) -> T {
        // TODO
        self.mean
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use math::Normalization;

    #[test]
    fn test_parameters() {
        let a = vec![1.0f32, 2.0, 3.0, 2.0, 1.0, 2.0];
        let mut p = Gaussian::estimate(&a, Normalization::N);
        assert!(p.mean - 1.8333 <= 0.0001);
        assert!(p.var - 0.47222 <= 0.00001);

        p = Gaussian::estimate(&a, Normalization::MinusOne);
        assert!(p.mean - 1.8333 <= 0.0001);
        assert!(p.var - 0.56667 <= 0.00001);
    }
}
