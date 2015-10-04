extern crate rand;

use matrix::Matrix;
use ops::{MatrixVectorOps, Functions, VectorVectorOps, MatrixScalarOps};
use vectors::{Append, random};
use ops_inplace::{MatrixMatrixOpsInPlace};
use opt::OptParams;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<usize>,
    params: Vec<Matrix<f64>>
}

/// Implementation of a neural network.
impl NeuralNetwork {

    /// Creates a new neural network.
    ///
    /// The network does not contain any layer. Layers can be added with the method
    /// `add_layer`.
    pub fn new() -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![],
            params: vec![]
        }
    }

    /// Creates random parameters which connect one layer with the other layer.
    ///
    /// The parameter `rows` denotes the number of layers in the right layer. The
    /// parameter `cols` denotes the number of layers in the left layer. If
    /// `from_input_layer` is `true` the left layer is an input layer. Otherwise,
    /// the left layer is a hidden layer.
    fn create_params(&self, rows: usize, cols: usize, from_input_layer: bool) -> Matrix<f64> {

        let n = if from_input_layer { cols } else { cols + 1 };

        Matrix::from_vec(random::<f64>(rows * n), rows, n).unwrap()
    }

    /// Adds an layer to the neural network.
    pub fn add_layer(&self, n: usize) -> NeuralNetwork {

        NeuralNetwork {
            layers: self.layers.append(&[n]),
            params: match self.layers.last() {
                Some(&m) => self.params.append(&[self.create_params(n, m, self.layers() == 1)]),
                None => vec![]
            }
        }
    }

    /// Sets the parameters which connect the given layer `layer` with the next
    /// layer.
    pub fn set_params(&self, layer: usize, params: Matrix<f64>) -> NeuralNetwork {

        let mut m = self.params.clone();

        match m.get_mut(layer) {
            None     => { panic!("Layer does not exist."); }
            Some(mx) => {
                assert!(mx.rows() == params.rows() && 
                    mx.cols() == params.cols(), "Parameter configuration is incompatible.");
                *mx = params;
            }
        }

        NeuralNetwork {
            layers: self.layers.clone(),
            params: m
        }
    }

    /// Returns the number of input units.
    pub fn input_size(&self) -> usize {

        assert!(self.layers.len() != 0, "No input layer defined.");
        *self.layers.first().unwrap()
    }

    /// Returns the number of output units.
    pub fn output_size(&self) -> usize {

        assert!(self.layers.len() != 0, "No output layer defined.");
        *self.layers.last().unwrap()
    }

    /// Returns the number of layers.
    pub fn layers(&self) -> usize {

        self.layers.len()
    }

    /// Computes the error of the net.
    pub fn error(&self, examples: &Matrix<f64>, targets: &Matrix<f64>) -> f64 {

        let mut err = 0.0;
        for (row, t) in examples.row_iter().zip(targets.row_iter()) {
            let v = self.predict(row).sub(t);
            err += v.iter().fold(0.0, |acc, val| acc + val * val);
        }
        err / (2.0 * examples.rows() as f64)
    }

    /// Computes the output of the neural network for the given input.
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(input.len() == self.input_size(), "Dimension of input vector does not match.");

        self.params.iter()
            .fold(input.to_vec(), 
                  |v, ref params| [1.0].append(&params.mul_vec(&v).sigmoid())
            )
            .iter().skip(1).cloned().collect() // skip the bias unit TODO more elegant way?
    }

    pub fn optimize(&self, x: &Matrix<f64>, labels: &Matrix<f64>, p: OptParams<f64>) -> NeuralNetwork {

        let a = p.alpha.unwrap();
        let mut n = self.clone();

        for _ in (0..p.iter.unwrap()) {
            let v = n.derivatives(x, labels).iter().map(|x| x.mul_scalar(-a)).collect::<Vec<_>>();
            n.update_params(&v);
        }
        n
    }

    fn feedforward(&self, x: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(x.len() == self.input_size(), "Dimension of input vector does not match.");

        let mut av = vec![x.to_vec()]; // inputs for the next layer (=sigmoid applied to outputs + bias unit)
        let mut zv = vec![x.to_vec()]; // outputs of previous layer without sigmoid
        let n = self.layers() - 2;

        for (idx, theta) in self.params.iter().enumerate() {
            let net = theta.mul_vec(&av.last().unwrap());
            if idx < n {
                av.push([1.0].append(&net.sigmoid()));
            } else {
                av.push(net.sigmoid());
            }
            zv.push(net);
        }
        (av, zv)
    }

    fn backprop(&self, output: &[f64], target: &[f64], av_zv: &(Vec<Vec<f64>>, Vec<Vec<f64>>)) -> Vec<Vec<f64>> {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(output.len() == self.output_size(), "Dimension of output vector does not match.");
        assert!(target.len() == self.output_size(), "Dimension of output vector does not match.");
        assert!(av_zv.0.len() == self.layers(), "Invalid dimension of vectors in av_zv.");
        assert!(av_zv.1.len() == self.layers(), "Invalid dimension of vectors in av_zv.");

        //let ref av = av_zv.0;
        let ref zv = av_zv.1;
        let mut deltas = vec![];

        let mut pos = (1..self.layers()).collect::<Vec<usize>>();
        
        // error of output layer
        let p = pos.pop().unwrap();
        deltas.push(output.sub(&target).mul(&zv[p].sigmoid_derivative()));

        // error of hidden layers
        while pos.len() > 0 {
            let p = pos.pop().unwrap();
            let mut v = self.params[p].transp_mul_vec(&deltas.last().unwrap());
            v.remove(0);
            deltas.push(v.mul(&zv[p].sigmoid_derivative()));
        }

        // the first entry is the delta vector for the output layer
        deltas
    }

    fn update(&self, acc: &mut Vec<Matrix<f64>>, deltas: &Vec<Vec<f64>>, a: &Vec<Vec<f64>>) {

        let mut dp = deltas.len();
        for i in (0..acc.len()) {
            dp -= 1;
            acc[i].iadd(&deltas[dp].col_mul_row(&a[i]));
        }
    }

    pub fn derivatives(&self, examples: &Matrix<f64>, targets: &Matrix<f64>) -> Vec<Matrix<f64>> {

        assert!(self.layers.len() >= 2, "At least two layers are required.");
        assert!(examples.rows() == targets.rows(), "Number of examples and labels mismatch.");
        assert!(examples.cols() == self.input_size(), "Dimension of input vector does not match.");
        assert!(self.output_size() == targets.cols(), "Dimension of target values mismatch.");

        // create accumulator for the deltas
        let mut acc_d = self.params.iter().map(|ref m| Matrix::fill(0.0, m.rows(), m.cols())).collect();

        // x = example
        // t = target vector
        for (x, t) in examples.row_iter().zip(targets.row_iter()) {

            let (av, zv) = self.feedforward(x);
            let deltas = self.backprop(&av.last().unwrap().clone(), t, &(av.clone(), zv));
            self.update(&mut acc_d, &deltas, &av);
        }

        for i in &mut acc_d {
            i.idiv_scalar(examples.rows() as f64);
        }
        acc_d
        // TODO tests
    }

    /// Updates the parameters of the network.
    ///
    /// Each matrix in `deltas` is added to the corresponding matrix
    /// of parameters of the network, i.e. the first matrix which
    /// contains the parameters from the first layer to the second layer,
    /// the second matrix is added to the parameters which contains the
    /// parameters from the second layer to the third layer and so on.
    pub fn update_params(&mut self, deltas: &[Matrix<f64>]) {

        assert!(self.params.len() == deltas.len(), "Dimensions do not match.");
        for i in (0..self.params.len()) {
            self.params[i].iadd(&deltas[i]);
        }
    }

    /// Returns the parameters of the network.
    pub fn params(&self) -> Vec<Matrix<f64>> {
        self.params.clone()
    }
}


#[cfg(test)]
mod tests {
    extern crate num;

    use self::num::abs;
    use super::*;
    use matrix::*;
    use ops::Functions;

    #[test]
    fn test_nn_create_params() {

        // 5 = number of units in new layer (rows in matrix)
        // 3 = number of units in last layer (columns in matrix)
        let a = NeuralNetwork::new().create_params(5, 3, true);
        assert_eq!(a.rows(), 5);
        assert_eq!(a.cols(), 3);
        let b = NeuralNetwork::new().create_params(5, 3, false);
        assert_eq!(b.rows(), 5);
        assert_eq!(b.cols(), 4);
    }

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
        assert_eq!(a.params[0].cols(), 4);

        let c = NeuralNetwork::new().add_layer(4).add_layer(6).add_layer(11);
        assert_eq!(c.layers, [4, 6, 11].to_vec());
        assert_eq!(c.params.len(), 2);
        assert!(c.params[0].rows() == 6 && c.params[0].cols() == 4);
        assert!(c.params[1].rows() == 11 && c.params[1].cols() == 7);
    }

    #[test]
    fn test_sigmoid() {

        assert!(vec![1.0, 2.0].sigmoid().similar(&vec![0.73106, 0.88080], 0.0001));
    }

    #[test]
    fn test_sigmoid_derivative() {

        let a = vec![1.0, 2.0];
        assert!(a.sigmoid_derivative().similar(&vec![0.19661, 0.10499], 0.00001));
    }

    #[test]
    fn test_set_params() {

        let m = mat![
            0.1, 0.2, 0.4;
            0.5, 2.0, 0.2
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
        let m = mat![0.1, 0.2, 0.4];

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
        println!("{:?}", p);
        assert!(p.similar(&vec![0.61301], 0.00001));
    }

    #[test]
    fn test_predict_three_layer() {

        // parameters
        let params1 = mat![
            0.1, 0.2, 0.4;
            0.2, 0.1, 2.0
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
        assert!(p.similar(&vec![0.88547], 0.00001));
    }

    #[test]
    fn test_feedforward() {

        // parameters
        let params1 = mat![
            0.1, 0.2, 0.4;
            0.2, 0.1, 2.0
        ];

        let params2 = mat![
            0.8, 1.2, 0.6;
            0.4, 0.5, 0.8;
            1.4, 1.5, 2.0
        ];

        let n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(2)
            .add_layer(3)
            .set_params(0, params1)
            .set_params(1, params2);

        let (a, z) = n.feedforward(&[0.5, 1.2, 1.5]);

        assert_eq!(a.len(), 3);
        assert_eq!(z.len(), 3);
        assert_eq!(z[0], vec![0.5, 1.2, 1.5]);
        assert_eq!(a[0], vec![0.5, 1.2, 1.5]);
        assert!(z[1].similar(&vec![0.89, 3.22], 0.001));
        assert!(a[1].similar(&vec![1.0, 0.70889, 0.96158], 0.00001));
        assert!(z[2].similar(&vec![2.2276, 1.5237, 4.3865], 0.0001));
        assert!(a[2].similar(&vec![0.90270, 0.82108, 0.98771], 0.00001));

        let d = n.backprop(&a[2].clone(), &[2.7, 3.1, 1.5], &(a, z));
        assert!(d[0].similar(&vec![-0.1578584, -0.3347843, -0.0062193], 0.0000002));
        assert!(d[1].similar(&vec![-0.075561, -0.013853], 0.000002));

        // TODO: test a 4 layer network
    }


    #[test]
    fn test_update() {

        let m1 = mat![
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0
        ];

        let m2 = mat![
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0
        ];

        let a1 = vec![0.4, 0.2, 0.3];
        let a2 = vec![0.7, 0.8, 0.2];

        let d3 = vec![0.6, 0.2, 0.5, 0.3];
        let d2 = vec![0.4, 0.1];

        let mut m = vec![m1, m2];
        let d = vec![d3, d2];
        let a = vec![a1, a2];

        let n = NeuralNetwork::new();
        n.update(&mut m, &d, &a);

        assert!(m[0].similar(&mat![
            0.16, 0.08, 0.12;
            0.04, 0.02, 0.03
        ], 0.01));

        assert!(m[1].similar(&mat![
            0.42, 0.48, 0.12;
            0.14, 0.16, 0.04;
            0.35, 0.40, 0.10;
            0.21, 0.24, 0.06
        ], 0.01));
    }

    #[test]
    fn test_errror() {

        // parameters
        let params1 = mat![
            0.1, 0.2, 0.4;
            0.2, 0.1, 2.0
        ];

        let params2 = mat![
            0.8, 1.2, 0.6;
            0.4, 0.5, 0.8;
            1.4, 1.5, 2.0
        ];

        let n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(2)
            .add_layer(3)
            .set_params(0, params1)
            .set_params(1, params2);

        let x = mat![
            0.5, 1.2, 1.5;
            1.0, 2.0, 1.0;
            3.0, 1.4, 4.2
        ];

        let t = mat![
            0.4, 1.0, 0.8;
            1.2, 0.4, 0.2;
            0.6, 0.3, 1.1
        ];
        let e = n.error(&x, &t);

        println!("er {}", e);
        assert!(num::abs(e - 0.26807) <= 0.0001);
    }

    #[test]
    fn test_update_params() {
        
        let params1 = mat![ 0.1, 0.2, 0.4; 0.2, 0.1, 2.0 ];
        let params2 = mat![ 0.8, 1.2, 0.6; 0.4, 0.5, 0.8;
            1.4, 1.5, 2.0
        ];

        let mut n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(2)
            .add_layer(3)
            .set_params(0, params1)
            .set_params(1, params2);

        let d1 = mat![ 0.5, 0.3, 0.2; 0.5, 0.9, 1.4 ];
        let d2 = mat![ 0.1, 1.0, 1.2; 0.1, 0.4, 1.1;
            1.0, 2.0, 3.0
        ];

        n.update_params(&[d1, d2]);
        assert!(n.params[0].similar(&mat![
            0.6, 0.5, 0.6; 0.7, 1.0, 3.4
        ], 0.001));
        assert!(n.params[1].similar(&mat![
            0.9, 2.2, 1.8; 0.5, 0.9, 1.9; 2.4, 3.5, 5.0
        ], 0.001));
    }

    #[test]
    fn test_params() {

        let params1 = mat![ 0.1, 0.2, 0.4; 0.2, 0.1, 2.0 ];
        let params2 = mat![ 0.8, 1.2, 0.6; 0.4, 0.5, 0.8;
            1.4, 1.5, 2.0
        ];
        let n = NeuralNetwork::new()
            .add_layer(3)
            .add_layer(2)
            .add_layer(3)
            .set_params(0, params1.clone())
            .set_params(1, params2.clone());
        let p = n.params();
        assert!(p[0].eq(&params1));
        assert!(p[1].eq(&params2));
    }
}

