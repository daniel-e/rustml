extern crate num;

use math::{Dimension, Normalization};
use math::Mean;

// ----------------------------------------------------------------------------

/// Trait to compute variance of values.
pub trait Var<T> {
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
                match dim {
                    Dimension::Row => {
                        match self.len() {
                            0 => 0.0 as $t,
                            n => {
                                let mu = self.mean(dim);
                                let mut d = n as $t;
                                match nrm {
                                    Normalization::MinusOne => {
                                        if n > 1 {
                                            d = d - (1.0 as $t);
                                        }
                                    }
                                    _ => ()
                                }
                                self.iter().map(|&x| (x - mu) * (x - mu)).fold(0.0 as $t, |acc, x| acc + x) / d
                            }
                        }
                    }
                    _ => 0.0 as $t
                }
            }
        }
    )*)
}

var_impl!{ f32 f64 }

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

