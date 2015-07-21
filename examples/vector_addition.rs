extern crate rustml;

use rustml::*;

fn main() {

    let mut v = vec![1.0, 2.0, 3.0];
    let y     = vec![3.0, 6.0, 9.0];
    v.iadd(&y);
    assert_eq!(v, vec![4.0, 8.0, 12.0]);
}
