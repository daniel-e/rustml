extern crate rustml;
extern crate num;

use rustml::octave::*;
use rustml::opencv::*;
use rustml::opt::*;
use num::pow;

fn main() {

    let fxy = |x: f64, y: f64| (pow(x, 2) + pow(y, 2)).sqrt();
    // objective function to be minimized
    let f   = |p: &[f64]| pow(fxy(p[0], p[1]) - 2.0, 2) + 10.0 * (fxy(p[0], p[1]) - 2.0).sin();
    // partial derivatives
    let fd  = |p: &[f64]| vec![
        2.0 * (fxy(p[0], p[1]) - 2.0) * p[0] / fxy(p[0], p[1]) + 10.0 * (fxy(p[0], p[1]) - 2.0).cos() * p[0] / fxy(p[0], p[1]),
        2.0 * (fxy(p[0], p[1]) - 2.0) * p[1] / fxy(p[0], p[1]) + 10.0 * (fxy(p[0], p[1]) - 2.0).cos() * p[1] / fxy(p[0], p[1])
    ];

    // set the number of iterations and the learning rate
    let opts = empty_opts()
        .iter(15)
        .alpha(0.04);

    let r1 = opt(&f, // objective to minimize
        &fd,         // derivatives
        &[2.7, 2.5], // initial parameters
        opts         // optimization options
    );

    // do a second optimization starting at a different location
    let r2 = opt(&f, &fd, &[-6.0, 6.0], opts);

    // ---- plot

    let data1 = r1.fvals.iter()
        .map(|&(ref params, val)| vec![params[0], params[1], val])
        .collect::<Vec<Vec<f64>>>();

    let data2 = r2.fvals.iter()
        .map(|&(ref params, val)| vec![params[0], params[1], val])
        .collect::<Vec<Vec<f64>>>();

    let o = builder()
        .add("x = linspace(-7, 7, 80); y = linspace(-7, 7, 80)")
        .add("[xx, yy] = meshgrid(x, y)")
        .add("r = sqrt(xx .^ 2 + yy .^ 2)")
        .add("z = (r-2) .^ 2 + 10 * sin(r-2)")
        .add("mesh(x, y, z)")
        .add("axis([-7,7,-7,7,-10,40])")
        .add("view(340, 65)")
        .add("hold on")
        .add("hidden off")
        .add("grid off")
        .add_values("plot3($1, $2, $3, 'linestyle', '-', 'marker', 'o', 'markerfacecolor', 'yellow', 'color', 'red', 'markersize', 5)", &data1)
        .add_values("plot3($1, $2, $3, 'linestyle', '-', 'marker', 'o', 'markerfacecolor', 'yellow', 'color', 'blue', 'markersize', 5)", &data2)
        .add("print -dpng /tmp/3dplot.png");

    o.run("/tmp/3dplot.m").unwrap();
    Window::new()
        .show_image(&RgbImage::from_file("/tmp/3dplot.png").unwrap())
        .wait_key();
}
