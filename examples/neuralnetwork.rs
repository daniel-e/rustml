#[macro_use] extern crate rustml;

use std::iter::repeat;

use rustml::nn::*;
use rustml::octave::builder;
use rustml::opencv::{Window, RgbImage};
use rustml::*;
use rustml::opt::empty_opts;

// This example trains a neural network to compute
// the XOR function.

fn main() {
    // create the dataset
    let seed = [1, 2, 3, 4];
    let n = 50;
    let x = mixture_builder()
            .add(n, normal_builder(seed).add(0.0, 0.2).add(0.0, 0.2))
            .add(n, normal_builder(seed).add(1.0, 0.2).add(1.0, 0.2))
            .add(n, normal_builder(seed).add(0.0, 0.2).add(1.0, 0.2))
            .add(n, normal_builder(seed).add(1.0, 0.2).add(0.0, 0.2))
            .as_matrix()
            .rm_column(0);

    // create the labels
    let labels = Matrix::from_it(
            repeat(0.0).take(2 * n).chain(repeat(1.0).take(2 * n)), 1
        );

    let n = NeuralNetwork::new()
        .add_layer(2)   // input layer with two units
        .add_layer(3)   // hidden layer with two units
        .add_layer(1)   // output layer
        .gd(&x, &labels, empty_opts().alpha(20.0).iter(500));
        
    // create the values for the contour plot
    let mut p = vec![];
    for y in (-1.0).linspace(2.0, 100) {
        for x in (-1.0).linspace(2.0, 100) {
            p.push(*n.predict(&[x, y].to_matrix(1)).get(0, 0).unwrap());
        }
    }

    builder()
        // contour plot
        .add_vector("P = $$", &p)
        .add("tx = ty = linspace(-1, 2, 100)")
        .add("contour(tx, ty, reshape(P, 100, 100))")
        .add("hold on")
        // examples
        .add_matrix("X = $$", &x)
        .add_vector_iter("y = $$", labels.values())
        .add("plot(X(y == 0, 1), X(y == 0, 2), 'd', 'markersize', 6, 'markerfacecolor', 'yellow', 'color', 'black')")
        .add("plot(X(y == 1, 1), X(y == 1, 2), 'o', 'markersize', 6, 'markerfacecolor', 'blue', 'color', 'black')")
        .add("axis([-1, 2, -1, 2])")
        .add("grid on")
        .add("print -dpng -r100 '/tmp/nn.png'")
        .run("/tmp/nn.m")
        .unwrap();

    Window::new().show_image(&RgbImage::from_file("/tmp/nn.png").unwrap()).wait_key();
}

