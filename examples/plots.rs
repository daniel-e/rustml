extern crate rustml;

use rustml::datasets::normal_builder;
use rustml::octave::builder;
use rustml::opencv::{Window, RgbImage};
use rustml::matrix::Matrix;

pub fn main() {

    let seed = [2, 3, 5, 7];

    let data1 = normal_builder(seed).add(1.0, 1.2).add(2.0, 1.2).take(100).collect::<Vec<Vec<f64>>>();
    let data2 = normal_builder(seed).add(5.0, 1.5).add(6.0, 1.5).take(100).collect::<Vec<Vec<f64>>>();
    let data3 = normal_builder(seed).add(6.0, 1.5).add(0.0, 1.5).take(100).collect::<Vec<Vec<f64>>>();

    let m1 = Matrix::from_row_vectors(&data1).unwrap();
    let m2 = Matrix::from_row_vectors(&data2).unwrap();
    let m3 = Matrix::from_row_vectors(&data3).unwrap();

    builder()
        .add_columns("plot($1, $2, 'o', 'markerfacecolor', 'yellow', 'color', 'black', 'markersize', 5)", &m1)
        .add("hold on")
        .add_columns("plot($1, $2, 'o', 'markerfacecolor', 'blue', 'color', 'black', 'markersize', 5)", &m2)
        .add_columns("plot($1, $2, 'o', 'markerfacecolor', 'red', 'color', 'black', 'markersize', 5)", &m3)
        .add("grid on")
        .add("axis([-2, 10, -5, 10])")
        .add("print -r50 -dpng /tmp/plot_normal.png")
        .run("/tmp/plot_normal.m")
        .unwrap();

    Window::new()
        .show_image(&RgbImage::from_file("/tmp/plot_normal.png").unwrap())
        .wait_key();
}

