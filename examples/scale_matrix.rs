#[macro_use] extern crate rustml;
extern crate num;

use rustml::Matrix;
use rustml::scaling::Scaling;
use rustml::gaussian::GaussianFunctions;

fn main() {
    let m = mat![
        1.0, 50.0, 1000.0;
        0.3, 45.0, 1200.0;
        1.9, 44.0, 1150.0;
        0.7, 60.0, 1300.0
    ];

    let (s, g) = m.scale();

    for (idx, i) in g.iter().enumerate() {
        println!("column {}: mean = {:4.2}, std = {:.2}", idx, i.mean(), i.std());
    }

    println!("\n{}", s);

    // In this case the all absolute values should be less than 1.5.
    assert!(
        s.buf().iter().all(|&x| num::abs(x) < 1.5)
    );
}

