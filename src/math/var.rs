extern crate num;

use self::num::traits::{Num, FromPrimitive};

use vectors::zero;
use math::{Dimension, Normalization};
use math::Mean;

// ----------------------------------------------------------------------------

/// Trait to compute variance of values.
pub trait Var<T> {
    fn var(&self, dim: Dimension, nrm: Normalization) -> T;
}

impl <T: Num + Copy + FromPrimitive> Var<T> for Vec<T> {

    fn var(&self, dim: Dimension, nrm: Normalization) -> T {
        (&self[..]).var(dim, nrm)
    }
}

impl <T: Num + Copy + FromPrimitive> Var<T> for [T] {

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
    use math::{Dimension, Normalization};

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
}

