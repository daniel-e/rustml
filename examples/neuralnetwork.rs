#[macro_use] extern crate rustml;

use rustml::nn::NeuralNetwork;
use rustml::octave::builder;
use rustml::datasets::{mixture_builder, normal_builder};
use rustml::opencv::{Window, RgbImage};
use rustml::matrix::Matrix;
use rustml::ops_inplace::MatrixMatrixOpsInPlace;

// This example trains a neural network to compute
// the XOR function.

fn compute_xor(r: &[f64]) -> f64 {

    let x = if r[1] > 0.5 { 1 } else { 0 };
    let y = if r[2] > 0.5 { 1 } else { 0 };
    if x != y {
        1.0
    } else {
        0.0
    }
}

fn main() {

    // create a random matrix where each element is either 0 or 1
    let seed = [1, 2, 3, 4];
    let n = 50;
    let examples = 
        mixture_builder()
            .add(n, normal_builder(seed).add(0.0, 0.1).add(0.0, 0.1))
            .add(n, normal_builder(seed).add(0.0, 0.1).add(1.0, 0.1))
            .add(n, normal_builder(seed).add(1.0, 0.1).add(1.0, 0.1))
            .add(n, normal_builder(seed).add(1.0, 0.1).add(0.0, 0.1))
            .as_matrix();

    let labels = examples
        .row_iter()
        .map(|r| compute_xor(r))
        .collect::<Vec<_>>();

    let mut n = NeuralNetwork::new()
        .add_layer(2)  // input layer with two units
        .add_layer(3)  // hidden layer with two units
        .add_layer(1);  // output layer

    for i in (0..400) {
        println!("{}", i);
        let mut d = n.derivatives(
            &examples.rm_column(0).unwrap(), 
            &Matrix::from_vec(labels.clone(), labels.len(), 1).unwrap()
        );
        for j in &mut d {
            j.imul_scalar(-0.1);
        }
        println!("{:?}", d[0]);
        n.update_weights(&d);
    }

    for row in examples.rm_column(0).unwrap().row_iter() {
        println!("{:?}", row);
        println!("{:?}", n.predict(row));
    }

    builder()
        .add_matrix("X = $$", &examples)
        .add_vector("y = $$", &labels)
        .add("plot(X(y == 0, 2), X(y == 0, 3), 'd', 'markersize', 6, 'markerfacecolor', 'yellow', 'color', 'black')")
        .add("hold on")
        .add("plot(X(y == 1, 2), X(y == 1, 3), 'o', 'markersize', 6, 'markerfacecolor', 'blue', 'color', 'black')")
        .add("axis([-1, 2, -1, 2])")
        .add("grid on")
        .add("print -dpng -r100 '/tmp/nn.png'")
        .run("/tmp/nn.m")
        .unwrap();

    Window::new().show_image(&RgbImage::from_file("/tmp/nn.png").unwrap()).wait_key();
}

