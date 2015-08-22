extern crate rustml;
extern crate num;

use rustml::opt::*;
use num::pow;

fn main() {

    // set the number of iterations to 10
    let opts = OptParams::new().iter(10);

    let r = opt(
        |p| pow(p[0] - 2.0, 2),       // objective to be minimized: (x-2)^2
        |p| vec![2.0 * (p[0] - 2.0)], // derivitive
        &[4.0],                       // initial parameters
        opts                          // optimization options
    );

    for (iter, i) in r.fvals.iter().enumerate() {
        println!("error after iteration {} was {}", iter + 1, i);
    }
    println!("solution: {:?}", r.params);
}
