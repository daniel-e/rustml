extern crate rand;

use matrix::Matrix;
use ops::{MatrixVectorOps, Functions, VectorVectorOps};
use vectors::{Append, random};

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<usize>,
    params: Vec<Matrix<f64>>
}

/// Implementation of a neural network.
impl NeuralNetwork {
    /// Creates a new neural network.
    pub fn new() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![],
            params: vec![]
        }
    }

    pub fn add_layer(&self, n: usize) -> NeuralNetwork {

        NeuralNetwork {
            layers: self.layers.append(&[n]),
            params: match self.layers.last() {
                Some(&m) => self.params.append(&[Matrix::from_vec(random::<f64>((m + 1) * n), n, m + 1).unwrap()]),
                None => vec![]
            }
        }
    }

    pub fn set_params(&self, layer: usize, params: Matrix<f64>) -> NeuralNetwork {

        let mut m = self.params.clone();

        match m.get_mut(layer) {
            None     => { panic!("Layer does not exist."); }
            Some(mx) => {
                assert!(mx.rows() == params.rows() && mx.cols() == params.cols(), "Parameter configuration is incompatible.");
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
                  |v, ref params| params.mul_vec(&[1.0].append(&v)).sigmoid()
            )
    }

    pub fn derivatives(&self, examples: &Matrix<f64>, targets: &Matrix<f64>) -> Vec<Matrix<f64>> {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(examples.rows() == targets.rows(), "Number of examples and labels mismatch.");
        assert!(examples.cols() == self.input_size(), "Dimension of input vector does not match.");
        assert!(self.output_size() == targets.cols(), "Dimension of target values mismatch.");

        // create accumulator for the deltas
        let mut acc_d = Vec::new();
        for m in &self.params {
            acc_d.push(Matrix::fill(0.0, m.rows(), m.cols()));
        }

        // TODO create the $\Delta$s

        for (x, t) in examples.row_iter().zip(targets.row_iter()) {
            // x = example
            // t = target vector

            // feedforward
            let mut av = vec![[1.0].append(x)]; // inputs for the next layer (=sigmoid applied to outputs + bias unit)
            let mut zv = vec![x.to_vec()];      // outputs of previous layer without sigmoid
            for theta in &self.params {
                let net = theta.mul_vec(&av.last().unwrap());
                zv.push(net.clone());
                av.push([1.0].append(&net.sigmoid()));
            }

            let mut deltas_per_layer = vec![];

            // deltas for output layer
            {
                let z = zv.pop().unwrap();
                deltas_per_layer.push(z.sigmoid().sub(t).mul(&z.sigmoid_derivative()));
            }

            let mut l = self.params.len() - 1;
            while zv.len() > 1 {
                let delta_next = deltas_per_layer.last().unwrap().clone(); // delta of layer l + 1
                let z = zv.pop().unwrap();
                let mut b = self.params[l].transp_mul_vec(&delta_next);
                b.remove(0);
                deltas_per_layer.push(b.mul(&z.sigmoid_derivative()));
                l = l - 1;
            }

            
            // TODO: add to the  $\Delta$s
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
    use ops::Functions;

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

        assert!(vec![1.0, 2.0].sigmoid().similar(&vec![0.73106, 0.88080], 0.0001));
    }

    #[test]
    fn test_sigmoid_derivative() {

        let a = vec![1.0, 2.0];
        assert!(a.sigmoid_derivative().similar(&vec![0.56683, 0.52977], 0.00001));
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

