extern crate rand;

use self::rand::Rng;
use matrix::Matrix;
use ops::MatrixVectorMul;

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<usize>,
    params: Vec<Matrix<f64>>
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![],
            params: vec![]
        }
    }

    pub fn add_layer(&self, n: usize) -> NeuralNetwork {

        match self.layers.last() {
            Some(&m) => {
                let mut rng = rand::thread_rng();
                let v = rng.gen_iter::<f64>().take((m + 1) * n).collect::<Vec<f64>>();
                let p = Matrix::from_vec(v, n, m + 1).unwrap();

                NeuralNetwork {
                    layers: self.layers.iter().chain([n].iter()).cloned().collect::<Vec<usize>>(),
                    params: self.params.iter().chain([p].iter()).cloned().collect::<Vec<Matrix<f64>>>()
                }
            }
            None => NeuralNetwork {
                layers: vec![n],
                params: vec![]
            }
        }
    }

    pub fn set_params(&self, layer: usize, params: Matrix<f64>) -> Option<NeuralNetwork> {

        let mut m = self.params.clone();

        match m.get_mut(layer) {
            None     => { return None; }
            Some(mx) => {
                if mx.rows() != params.rows() || mx.cols() != params.cols() {
                    return None;
                }
                *mx = params;
            }
        }

        Some(NeuralNetwork {
            layers: self.layers.clone(),
            params: m
        })
    }

    pub fn output(&self, input: &[f64]) -> Option<Vec<f64>> {

        // at least two layers are required and the length
        // of the input vector must be equal to the number
        // of units in the first layer
        if self.layers.len() < 2 || input.len() != self.layers.first.unwrap() {
            return None;
        }

        // TODO
        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

    #[test]
    fn test_nn() {

        let n = NeuralNetwork::new();
        assert_eq!(n.layers.len(), 0);
        assert_eq!(n.params.len(), 0);

        let b = NeuralNetwork::new().add_layer(3);
        assert_eq!(b.layers, [3].to_vec());
        assert_eq!(b.params.len(), 0);

        let a = NeuralNetwork::new().add_layer(4).add_layer(3);
        assert_eq!(a.layers, [4, 3].to_vec());
        assert_eq!(a.params.len(), 1);
        assert_eq!(a.params[0].rows(), 3);
        assert_eq!(a.params[0].cols(), 5);
    }

    #[test]
    fn test_set_params() {

        let m = mat![
            0.1, 0.2, 0.4, 0.5;
            0.5, 2.0, 0.2, 0.4
        ];

        let n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(2)
            .set_params(0, m.clone())
            .unwrap();

        assert!(n.params[0].eq(&m));
    }
}

