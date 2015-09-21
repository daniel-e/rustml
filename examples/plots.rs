extern crate rustml;

use rustml::datasets::{normal_builder, mixture_builder};
use rustml::octave::builder;
use rustml::opencv::{Window, RgbImage};
use rustml::matrix::Matrix;
use rustml::knn::classify;
use rustml::sliding;
use rustml::distance::{Distance, Euclid};

fn plot_normal_data() {

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

    Window::new()
        .show_image(&RgbImage::from_file("/tmp/plot_normal_1.png").unwrap())
        .wait_key();
}

pub fn plot_mixture() {

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

    Window::new()
        .show_image(&RgbImage::from_file("/tmp/plot_mixture.png").unwrap())
        .wait_key();

}

pub fn plot_knn_decision_boundary() {

    let seed = [2, 3, 5, 7];

    let m = 
        mixture_builder()
            .add(100, normal_builder(seed).add(1.0, 2.2).add(2.0, 1.2))
            .add(100, normal_builder(seed).add(5.0, 2.5).add(6.0, 2.5))
            .add(100, normal_builder(seed).add(6.0, 2.5).add(0.0, 2.5))
            .as_matrix();

    let labels = m.column(0).unwrap().iter().map(|&x| x.clone() as usize).collect::<Vec<usize>>();
    let mx = m.rm_column(0).unwrap();

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

    Window::new()
        .show_image(&RgbImage::from_file("/tmp/plot_knn_boundary.png").unwrap())
        .wait_key();
}

pub fn main() {

    //plot_mixture();
    //plot_normal_data();
    plot_knn_decision_boundary();
}

