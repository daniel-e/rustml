extern crate rustml;

use std::process;
use std::env;
use rustml::opencv::*;
use rustml::consts::*;

fn err() {
    println!("\n\nThis example requires the Rustml datasets package.");
    println!("Please have a look at https://github.com/daniel-e/rustml#rustml-datasets-package");
    println!("for details on how to install it.\n");
}

fn main() {
    let waitkey = env::args().skip(1).next().unwrap_or("".to_string()) != "--nokey".to_string();

    match RgbImage::from_file(&path_for("images/fog.jpg").unwrap()) {
        None => {
            err();
            process::exit(0);
        }
        Some(i) => {
            let j = i.resize(100, 100);

            if waitkey {
                println!("Original image. Press any key to continue.");
                Window::new().show_image(&i).wait_key();
                println!("Resized image. Press any key to quit.");
                Window::new().show_image(&j).wait_key();
            }
        }
    }
}
