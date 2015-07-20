use gaussian::Gaussian;
use matrix::Matrix;
use math::{Dimension, Normalization, Mean, Var};

pub trait Scaling<T> {
    fn scale(&self) -> (Self, Vec<Gaussian<T>>);
}


macro_rules! scaling_impl {
    ($($t:ty)*) => ($(

        impl Scaling<$t> for Matrix<$t> {

            fn scale(&self) -> (Matrix<$t>, Vec<Gaussian<$t>>) {

                let mean_vec = self.mean(Dimension::Column);
                let var_vec = self.var(Dimension::Column, Normalization::MinusOne);

                let r = mean_vec.iter().zip(var_vec.iter()).map(|(&x, &y)| Gaussian::new(x, y)).collect();

                let mut mr = Matrix::<$t>::fill(0 as $t, self.rows(), self.cols());
                

                // TODO
                (self.clone(), r)
            }
        }

    )*)
}

scaling_impl!{ f32 f64 }


