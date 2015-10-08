extern crate rustml;

use std::process;
use std::io::Write;
use std::fs::File;

use rustml::*;
use rustml::opencv::*;
use rustml::consts::*;

// note: to be able to run this example you need the rustml dataset package
// see https://github.com/daniel-e/rustml#datasets
//
// The result of this example can be plotted with Octave as follows:
// start Octave and type:
// > load("/tmp/day_and_night.txt");
// > hist(day_and_night, 256);

fn err() {
    println!("\n\nThis example requires the Rustml datasets package.");
    println!("Please have a look at https://github.com/daniel-e/rustml#rustml-datasets-package");
    println!("for details on how to install it.\n");
}

fn main() {

    let video = match Video::from_file(&path_for("videos/day_and_night_in_gray.avi").unwrap()) {
        None => {
            err();
            process::exit(0);
        },
        Some(x) => x
    };

    let mask = match GrayImage::from_file(&consts::path_for("videos/day_and_night_in_gray_mask_sky.png").unwrap()) {
        None => {
            err();
            process::exit(0);
        },
        Some(x) => x
    };

    let mut f = File::create("/tmp/day_and_night.txt").unwrap();

    for (idx, img) in video.gray_frame_iter().enumerate() {

        // write status to stderr for the impatient
        if idx % 1000 == 0 {
            writeln!(&mut std::io::stderr(), "{} frames", idx).unwrap();
        }

        writeln!(f, "{}",
            img.mask_iter(&mask).map(|x| x as u32).collect::<Vec<u32>>().mean()
        ).unwrap();
    }
}

