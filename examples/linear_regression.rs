extern crate rustml;
extern crate getopts;

use std::env;
use rustml::csv::from_csv_file;
use rustml::octave::builder;
use rustml::opencv::{Window, RgbImage};
use rustml::regression::{Hypothesis, DesignMatrix};
use rustml::opt::{plot_learning_curve, opt_hypothesis, empty_opts};

fn main() {
    let waitkey = env::args().skip(1).next().unwrap_or("".to_string()) != "--nokey".to_string();

    let dm = from_csv_file("datasets/testing/points.txt", " ")
        .unwrap().design_matrix();
    
    let y = dm.column(2).unwrap();
    let x = dm.rm_column(2).unwrap();

    let result = opt_hypothesis(
        &Hypothesis::from_params(&[0.7, 0.5]),
        &x, &y,
        empty_opts().alpha(0.05).iter(400)
    );

    let w = Window::new();

    if waitkey {
        plot_learning_curve(&result, &w).unwrap();
        w.wait_key();
    }

    // plot the hypothesis with the data points
    builder()
        .add("x = [0, 1]")
        .add_values("y = $1 + $2 * x", &result.params)
        .add("plot(x, y, 'linewidth', 2, 'color', 'red')")
        .add("hold on")
        .add_vector("x = $$", &x.column(1).unwrap())
        .add_vector("y = $$", &y)
        .add("plot(x, y, 'o')")
        .add("grid on")
        .add("axis('nolabel')")
        .add("print -r50 -dpng '/tmp/linreg_plot.png'")
        .run("/tmp/plot_script.m")
        .unwrap();

    if waitkey {
        w.show_image(&RgbImage::from_file("/tmp/linreg_plot.png").unwrap());
        w.wait_key();
    }
}

