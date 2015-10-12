extern crate rustml;
extern crate getopts;

use std::env;
use std::iter::repeat;

use rustml::octave::builder;
use rustml::opencv::{Window, RgbImage};
use rustml::knn::classify;
use rustml::nn::*;
use rustml::opt::empty_opts;
use rustml::*;

static mut WAITKEY: bool = true;

fn wait_key() -> bool { unsafe { WAITKEY } }

fn plot_normal_data() {

    println!("normal data ...");

    let seed = [1, 3, 5, 7];
    let data = normal_builder(seed).add(1.0, 0.3).add(2.0, 0.4).take(1000).collect::<Vec<Vec<f64>>>();
    let m = Matrix::from_row_vectors(&data).unwrap();

    builder()
        .add_columns("plot($1, $2, 'o', 'markerfacecolor', 'yellow', 'color', 'black', 'markersize', 5)", &m)
        .add("grid on")
        .add("axis([0, 2, 1, 3])")
        .add("axis('nolabel')")
        .add("print -r50 -dpng /tmp/plot_normal_1.png")
        .run("/tmp/plot_normal_1.m")
        .unwrap();

    if wait_key() {
        Window::new()
            .show_image(&RgbImage::from_file("/tmp/plot_normal_1.png").unwrap())
            .wait_key();
    }
}

pub fn plot_mixture() {

    println!("normal mixture ...");

    let seed = [2, 3, 5, 7];

    let m = 
        mixture_builder()
            .add(100, normal_builder(seed).add(1.0, 1.2).add(2.0, 1.2))
            .add(100, normal_builder(seed).add(5.0, 1.5).add(6.0, 1.5))
            .add(100, normal_builder(seed).add(6.0, 1.5).add(0.0, 1.5))
            .as_matrix();

    builder()
        .add_matrix("X = $$", &m)
        .add("A = X(X(:,1) == 0, 2:end)")
        .add("B = X(X(:,1) == 1, 2:end)")
        .add("C = X(X(:,1) == 2, 2:end)")
        .add("plot(A(:,1), A(:,2), 'o', 'markerfacecolor', 'yellow', 'color', 'black', 'markersize', 5)")
        .add("hold on")
        .add("plot(B(:,1), B(:,2), 'o', 'markerfacecolor', 'blue', 'color', 'black', 'markersize', 5)")
        .add("plot(C(:,1), C(:,2), 'o', 'markerfacecolor', 'red', 'color', 'black', 'markersize', 5)")
        .add("grid on")
        .add("axis([-2, 10, -5, 10])")
        .add("axis('nolabel')")
        .add("print -r50 -dpng /tmp/plot_mixture.png")
        .run("/tmp/plot_normal.m")
        .unwrap();

    if wait_key() {
        Window::new()
            .show_image(&RgbImage::from_file("/tmp/plot_mixture.png").unwrap())
            .wait_key();
    }
}

pub fn plot_knn_decision_boundary() {

    println!("knn decision boundary ...");

    let seed = [2, 3, 5, 7];

    let m = 
        mixture_builder()
            .add(100, normal_builder(seed).add(1.0, 2.2).add(2.0, 1.2))
            .add(100, normal_builder(seed).add(5.0, 2.5).add(6.0, 2.5))
            .add(100, normal_builder(seed).add(6.0, 2.5).add(0.0, 2.5))
            .as_matrix();

    let labels = m.column(0).unwrap().iter().map(|&x| x.clone() as usize).collect::<Vec<usize>>();
    let mx = m.rm_column(0);

    let mut mt = Matrix::<f64>::new();
    for y in (-49..100) {
        for x in (-19..100) {
            let xp = x as f64 / 10.0;
            let yp = y as f64 / 10.0;
            let l = classify(&mx, &labels, &[xp, yp], 5, |x, y| Euclid::compute(x, y).unwrap());
            mt.add_row(&[l as f64, xp, yp]);
        }
    }

    builder()
        .add_matrix("X = $$", &mt)
        .add("A = X(X(:,1) == 0, 2:end)")
        .add("B = X(X(:,1) == 1, 2:end)")
        .add("C = X(X(:,1) == 2, 2:end)")
        .add("scatter(A(:,1), A(:,2), 5, [1, 1, 0.7], 'filled')")
        .add("hold on")
        .add("scatter(B(:,1), B(:,2), 5, [0.7, 0.7, 1], 'filled')")
        .add("scatter(C(:,1), C(:,2), 5, [1, 0.7, 0.7], 'filled')")
        .add_matrix("X = $$", &m)
        .add("A = X(X(:,1) == 0, 2:end)")
        .add("B = X(X(:,1) == 1, 2:end)")
        .add("C = X(X(:,1) == 2, 2:end)")
        .add("plot(A(:,1), A(:,2), 'o', 'markerfacecolor', 'yellow', 'color', 'black', 'markersize', 7)")
        .add("plot(B(:,1), B(:,2), 's', 'markerfacecolor', 'blue', 'color', 'black', 'markersize', 6)")
        .add("plot(C(:,1), C(:,2), 'd', 'markerfacecolor', 'red', 'color', 'black', 'markersize', 8)")
        .add("grid on")
        .add("axis([-2, 10, -5, 10])")
        .add("axis('nolabel')")
        .add("print -r50 -dpng /tmp/plot_knn_boundary.png")
        .run("/tmp/plot_knn_boundary.m")
        .unwrap();

    if wait_key() {
        Window::new()
            .show_image(&RgbImage::from_file("/tmp/plot_knn_boundary.png").unwrap())
            .wait_key();
    }
}

pub fn plot_nn() {

    println!("nn ...");

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
        ).unwrap();

    let n = NeuralNetwork::new()
        .add_layer(2)   // input layer with two units
        .add_layer(3)   // hidden layer with two units
        .add_layer(1)   // output layer
        .gd(&x, &labels, empty_opts().alpha(20.0).iter(500));
        
    // create the values for the contour plot
    let mut p = vec![];
    for y in (-1.0).linspace(2.0, 100) {
        for x in (-1.0).linspace(2.0, 100) {
            p.push(*n.predict(&[x, y].to_matrix()).row(0).unwrap().first().unwrap());
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
        .add("axis('nolabel')")
        .add("grid on")
        .add("print -dpng -r50 '/tmp/nn.png'")
        .run("/tmp/nn.m")
        .unwrap();
}

pub fn plot_nn_example() {

    println!("nn example ...");

    let seed = [1, 2, 3, 4];
    let n = 50;
    let x = mixture_builder()
            .add(n, normal_builder(seed).add(0.0, 0.5).add(0.0, 0.5))
            .add(n, normal_builder(seed).add(1.0, 0.5).add(1.0, 0.5))
            .as_matrix()
            .rm_column(0);

    // create the labels
    let labels = Matrix::from_it(
            repeat(0.0).take(n).chain(repeat(1.0).take(n)), 1
        ).unwrap();

    let n = NeuralNetwork::new()
        .add_layer(2)   // input layer with two units
        .add_layer(3)   // hidden layer
        .add_layer(1)   // output layer
        .gd(&x, &labels, empty_opts().alpha(20.0).iter(500));
        
    // create the values for the contour plot
    let mut p = vec![];
    for y in (-1.0).linspace(2.0, 100) {
        for x in (-1.0).linspace(2.0, 100) {
            p.push(*n.predict(&[x, y].to_matrix()).row(0).unwrap().first().unwrap());
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
        .add("axis('nolabel')")
        .add("grid on")
        .add("print -dpng -r50 '/tmp/nn_example.png'")
        .run("/tmp/nn_example.m")
        .unwrap();

    if wait_key() {
        Window::new()
            .show_image(&RgbImage::from_file("/tmp/nn_example.png").unwrap())
            .wait_key();
    }
}


pub fn main() {
    unsafe { WAITKEY = env::args().skip(1).next().unwrap_or("".to_string()) != "--nokey".to_string(); }

    plot_mixture();
    plot_normal_data();
    plot_knn_decision_boundary();
    plot_nn();
    plot_nn_example();
}

