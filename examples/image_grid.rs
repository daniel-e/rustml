extern crate rustml;
extern crate getopts;

use std::process;
use std::env;
use rustml::datasets::MnistDigits;
use rustml::opencv::{GrayImage, Image, Window};

fn main() {
    let waitkey = env::args().skip(1).next().unwrap_or("".to_string()) != "--nokey".to_string();


    let test = match MnistDigits::default_test_set() {
        Err(_) => {
            println!("\n\nThis example requires the MNIST dataset of handwritten digits.");
            println!("Please have a look at https://github.com/daniel-e/rustml#rustml-datasets-package");
            println!("for details on how to install it.\n");
            process::exit(0);
        }
        Ok(pair) => pair.0
    };

    let images = test
        .map(|&x| 255 - x) // invert the data
        .row_iter()
        .take(100)
        .map(|r| GrayImage::from_slice(r, 28, 28).unwrap())
        .collect::<Vec<GrayImage>>();

    let grid = GrayImage::grid(&images, 10, 0).unwrap();
    
    grid.to_file("/tmp/grid.png");

    println!("The image has been saved in /tmp/grid.png.");

    if waitkey {
        println!("Press any key to quit.");
        Window::new().show_image(&grid).wait_key();
    }
}
