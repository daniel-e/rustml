extern crate rustml;

use rustml::octave::*;
use rustml::opencv::*;

fn main() {
    let o = builder()
        .add("x = linspace(-10, 10, 80)")
        .add("y = x.^2 + sin(x) .* (x .* x)")
        .add("plot(x, y, 'linewidth', 2)")
        .add("grid on")
        .add("print -dpng '/tmp/example_plot.png'");

    o.run("/tmp/example_plot.m").unwrap();
    Window::new()
        .show_image(&RgbImage::from_file("/tmp/example_plot.png").unwrap())
        .wait_key();
}

