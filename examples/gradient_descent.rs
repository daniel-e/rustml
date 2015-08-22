extern crate rustml;

use rustml::opt::*;

fn main() {

    let opts = OptParams::new().iter(10);

    let r = opt(
        |x| (x[0] - 2.0) * (x[0] - 2.0), 
        |x| vec![2.0 * (x[0] - 2.0)], 
        &[4.0], 
        opts
    );

    for (iter, i) in r.fvals.iter().enumerate() {
        println!("error after iteration {} was {}", iter + 1, i);
    }
    println!("{:?}", r.params);
}
