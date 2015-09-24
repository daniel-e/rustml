extern crate rand;

use self::rand::Rng;
use matrix::Matrix;
use ops::MatrixVectorMul;
use vectors::Append;

pub fn sigmoid(v: &[f64]) -> Vec<f64> {

    v.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect::<Vec<f64>>()
}

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

    pub fn set_params(&self, layer: usize, params: Matrix<f64>) -> NeuralNetwork {

        let mut m = self.params.clone();

        match m.get_mut(layer) {
            None     => {
                panic!("Layer does not exist."); 
            }
            Some(mx) => {
                if mx.rows() != params.rows() || mx.cols() != params.cols() {
                    panic!("Number of parameters are incompatible.");
                }
                *mx = params;
            }
        }

        NeuralNetwork {
            layers: self.layers.clone(),
            params: m
        }
    }

    pub fn input_size(&self) -> usize {
        assert!(self.layers.len() != 0, "No input layer defined.");
        *self.layers.first().unwrap()
    }

    pub fn output_size(&self) -> usize {
        assert!(self.layers.len() != 0, "No output layer defined.");
        *self.layers.last().unwrap()
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(input.len() == self.input_size(), "Dimension of input vector does not match.");

        self.params.iter()
            .fold(input.to_vec(), 
                  |v, ref params| sigmoid(&params.mul_vec(&[1.0].append(&v)))
            )
    }

    pub fn derivatives(&self, examples: &Matrix<f64>, targets: &Matrix<f64>) -> Vec<Matrix<f64>> {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(examples.rows() == targets.rows(), "Number of examples and labels mismatch.");
        assert!(examples.cols() == self.input_size(), "Dimension of input vector does not match.");
        assert!(self.output_size() == targets.rows(), "Dimension of target values mismatch.");

        // create accumulator for the deltas
        let mut acc_d: Vec<Matrix<f64>> = Vec::new();
        for m in &self.params {
            acc_d.push(Matrix::fill(0.0, m.rows(), m.cols()));
        }

        for (x, t) in examples.row_iter().zip(targets.row_iter()) {
            // x = example
            // t = target vector

            // feedforward
            let mut a = vec![x.to_vec()]; // a^1 (=input vector)
            for theta in &self.params {
                let v = [1.0].append(&a.last().unwrap());
                a.push(sigmoid(&theta.mul_vec(&v)));
            }

            // delta for the output layer
            
            // TODO
        }

        acc_d
    }
}


#[cfg(test)]
mod tests {
    extern crate num;

    use self::num::*;
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
    fn test_sigmoid() {

        let v = [1.0, 2.0];
        let s = sigmoid(&v);
        assert!(num::abs(s[0] - 0.73106) < 0.0001);
        assert!(num::abs(s[1] - 0.88080) < 0.0001);
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
            .set_params(0, m.clone());

        assert_eq!(n.layers(), 2);
        assert_eq!(n.input_size(), 3);
        assert_eq!(n.output_size(), 2);

        assert!(n.params[0].eq(&m));
    }

    #[test]
    fn test_predict_two_layer() {

        // set parameters
        let m = mat![0.1, 0.2, 0.4, 0.5];

        // input vector
        let x = [0.4, 0.5, 0.8];

        let n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(1)
            .set_params(0, m);

        assert_eq!(n.layers(), 2);
        assert_eq!(n.input_size(), 3);
        assert_eq!(n.output_size(), 1);

        let p = n.predict(&x);
        assert_eq!(p.len(), 1);
        assert!(num::abs(p[0] - 0.68568) <= 0.00001);
    }

    #[test]
    fn test_predict_three_layer() {

        // parameters
        let params1 = mat![
            0.1, 0.2, 0.4, 0.5;
            0.2, 0.1, 2.0, 1.4
        ];

        let params2 = mat![
            0.8, 1.2, 0.6
        ];

        // input vector
        let x = [0.4, 0.5, 0.8];

        let n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(2)
            .add_layer(1)
            .set_params(0, params1)
            .set_params(1, params2);

        assert_eq!(n.layers(), 3);
        assert_eq!(n.input_size(), 3);
        assert_eq!(n.output_size(), 1);

        let p = n.predict(&x);
        assert_eq!(p.len(), 1);
        assert!(num::abs(p[0] - 0.89762) <= 0.00001);
    }
}

