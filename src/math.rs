//! Module with a collection of different mathematical functions.
//!

use matrix::Matrix;
use vectors::AddVector;

pub enum Dimension {
    Row,
    Column
}

/// Trait to compute the mean of values.
pub trait Mean<T> {
    fn mean(&self, dim: Dimension) -> Option<T>;
}

impl Mean<f32> for Vec<f32> {

    fn mean(&self, dim: Dimension) -> Option<f32> {
        
        match dim {
            Dimension::Row => {
                let mut i = self.iter();
                match i.next() {
                    None => None,
                    Some(init) => Some(
                        i.fold(*init, |init, val| init + val) / (self.len() as f32)
                    )
                }
            }
            _ => None
        }
    }
}

impl Mean<Vec<f32>> for Matrix<f32> {

    fn mean(&self, dim: Dimension) -> Option<Vec<f32>> {

        match dim {
            Dimension::Column => {
                let mut r: Vec<f32> = self.values().take(self.cols()).cloned().collect();
                for row in self.row_iter_at(1) {
                    r.add(row);
                }
                // TODO divide by n
                Some(r)
            }
            Dimension::Row => {
                None
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

    #[test]
    fn test_mean_vec_f32() {

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(x.mean(Dimension::Row).unwrap(), 2.5);

        let a: Vec<f32> = Vec::new();
        assert!(a.mean(Dimension::Row).is_none());
    }

    #[test]
    fn test_mean_mat_f32() {

        let x = mat![
            1.0, 2.0;
            3.0, 4.0
        ];

        // TODO
        //assert_eq!(x.mean(Dimension::Column).unwrap(), vec![2.0, 3.0]);
    }
}



    /*
    pub fn mean_cols(&self) -> Vec<T> {

        let mut r: Vec<T> = self.data.iter().take(self.cols()).cloned().collect();
        for row in self.row_iter_at(1) {
            r.add(row);
        }
        r
    }
    */


