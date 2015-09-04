extern crate rustml;

use rustml::csv::from_csv_file;
use rustml::octave::builder;
use rustml::opencv::{Window, RgbImage};
use rustml::regression::{Hypothesis, design_matrix};
use rustml::opt::{plot_learning_curve, opt_hypothesis, empty_opts};

fn main() {

    let dm = design_matrix(
        &from_csv_file("datasets/testing/points.txt", " ").unwrap()
    );
    let y = dm.column(2).unwrap();
    let x = dm.rm_column(2).unwrap();

    let result = opt_hypothesis(
        &Hypothesis::from_params(&[0.7, 0.5]),
        &x, &y,
        empty_opts().alpha(0.05).iter(400)
    );

    let w = Window::new();

    plot_learning_curve(&result, &w).unwrap();
    w.wait_key();

    // plot the hypothesis with the data points
    builder()
        .add("x = [0, 1]")
        .add_vals("y = $1 + $2 * x", &result.params)
        .add("plot(x, y, 'linewidth', 2, 'color', 'red')")
        .add("hold on")
        .add_vector("x = $$", &x.column(1).unwrap())
        .add_vector("y = $$", &y)
        .add("plot(x, y, 'o')")
        .add("grid on")
        .add("print -r100 -dpng '/tmp/linreg_plot.png'")
        .run("/tmp/plot_script.m")
        .unwrap();

    w.show_image(&RgbImage::from_file("/tmp/linreg_plot.png").unwrap());
    w.wait_key();
}

