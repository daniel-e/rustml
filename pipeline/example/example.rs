extern crate rand;

use rand::Rng;
use std::io::Write;
use std::fs::File;

extern crate pipeline;

fn main() {
    let c = pipeline::read_config().unwrap();

    let mut output = File::create(c.target).unwrap();
    let min = c.params.get("min").unwrap().parse::<usize>().unwrap();
    let max = c.params.get("max").unwrap().parse::<usize>().unwrap();
    let mut n = c.params.get("n").unwrap().parse::<usize>().unwrap();

    let mut rng = rand::thread_rng();
    while n > 0 {
        writeln!(output, "{}", rng.gen_range(min, max + 1)).unwrap();
        n -= 1;
    }
}

