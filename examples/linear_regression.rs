extern crate rustml;

use rustml::csv::*;
use rustml::octave::*;
use rustml::opencv::*;
use rustml::regression::*;
use rustml::opt::*;

fn main() {

    let dm = design_matrix(
        &from_csv_file("datasets/testing/points.txt", " ").unwrap()
    );
    let y = dm.column(2).unwrap();
    let x = dm.rm_column(2).unwrap();

    let h = Hypothesis::from_params(
        &opt_hypothesis(
            &Hypothesis::random(x.cols()),
            &x, &y,
            empty_opts()
        ).params
    );

    let p = h.params();

    builder()
        .add("x = [0, 1]")
        .add_vals("y = $1 + $2 * x", &p)
        .add("plot(x, y, 'linewidth', 2, 'color', 'red')")
        .add("hold on")
        .add_vector("x = $$", &x.column(1).unwrap())
        .add_vector("y = $$", &y)
        .add("plot(x, y, 'o')")
        .add("print -dpng '/tmp/linreg_plot.png'")
        .run("/tmp/plot_script.m")
        .unwrap();

    Window::new()
        .show_image(&RgbImage::from_file("/tmp/linreg_plot.png").unwrap())
        .wait_key(0);
}

