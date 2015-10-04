#[macro_use] extern crate rustml;

use std::iter::repeat;

use rustml::nn::NeuralNetwork;
use rustml::octave::builder;
use rustml::datasets::{mixture_builder, normal_builder};
use rustml::opencv::{Window, RgbImage};
use rustml::matrix::Matrix;
use rustml::ops_inplace::MatrixMatrixOpsInPlace;
use rustml::vectors::Linspace;

// This example trains a neural network to compute
// the XOR function.

fn main() {

    // create a random matrix where each element is either 0 or 1
    let seed = [1, 2, 3, 4];
    let n = 50;
    let examples = 
        mixture_builder()
            .add(n, normal_builder(seed).add(0.0, 0.2).add(0.0, 0.2))
            .add(n, normal_builder(seed).add(1.0, 0.2).add(1.0, 0.2))
            .add(n, normal_builder(seed).add(0.0, 0.2).add(1.0, 0.2))
            .add(n, normal_builder(seed).add(1.0, 0.2).add(0.0, 0.2))
            .as_matrix();

    let labels = repeat(0.0).take(2 * n).chain(repeat(1.0).take(2 * n))
        .collect::<Vec<_>>();

    let mut n = NeuralNetwork::new()
        .add_layer(2)   // input layer with two units
        .add_layer(3)   // hidden layer with two units
        .add_layer(1);  // output layer

    let x = examples.rm_column(0).unwrap();
    let y = Matrix::from_vec(labels.clone(), labels.len(), 1).unwrap();

    for _ in (0..500) {
        let mut d = n.derivatives(&x, &y);
        for j in &mut d { // alpha
            j.imul_scalar(-20.0);
        }
        n.update_weights(&d);
    }

    // create the matrix for the contour plot
    let mut p = Matrix::<f64>::new();
    for y in (-1.0).linspace(2.0, 90) {
        for x in (-1.0).linspace(2.0, 90) {
            p.add_row(&[x, y, *n.predict(&[x, y]).first().unwrap()]);
        }
    }

    builder()
        // contour plot
        .add_matrix("P = $$", &p)
        .add("tx = ty = linspace(-1, 1.96666666, 90)")
        .add("contour(tx, ty, reshape(P(:,3), 90, 90))")
        .add("hold on")
        // examples
        .add_matrix("X = $$", &examples)
        .add_vector("y = $$", &labels)
        .add("plot(X(y == 0, 2), X(y == 0, 3), 'd', 'markersize', 6, 'markerfacecolor', 'yellow', 'color', 'black')")
        .add("plot(X(y == 1, 2), X(y == 1, 3), 'o', 'markersize', 6, 'markerfacecolor', 'blue', 'color', 'black')")
        .add("axis([-1, 2, -1, 2])")
        .add("grid on")
        .add("print -dpng -r100 '/tmp/nn.png'")
        .run("/tmp/nn.m")
        .unwrap();

    Window::new().show_image(&RgbImage::from_file("/tmp/nn.png").unwrap()).wait_key();
}

