extern crate rustml;

use rustml::datasets::MnistDigits;
use rustml::opencv::{GrayImage, Image, Window};

fn main() {
    let test = MnistDigits::default_test_set().unwrap().0;

    let images = test
        .map(|&x| 255 - x) // invert the data
        .row_iter()
        .take(100)
        .map(|r| GrayImage::from_slice(r, 28, 28).unwrap())
        .collect::<Vec<GrayImage>>();

    let grid = GrayImage::grid(&images, 10, 0).unwrap();
    
    grid.to_file("/tmp/grid.png");

    println!("The image has been save in /tmp/grid.png.");
    println!("Press any key to quit.");

    Window::new().show_image(&grid).wait_key();
}
