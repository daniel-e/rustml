extern crate rustml;

use std::io::Write;
use rustml::*;
use rustml::opencv::*;

fn main() {
    // TODO example files
    let video = Video::from_file("/home/dz/videos/generated/reduced.avi").unwrap();
    let mask = GrayImage::from_file("/home/dz/videos/generated/mask.png").unwrap();

    for (idx, img) in video.gray_frame_iter().enumerate() {

        if idx % 1000 == 0 {
            writeln!(&mut std::io::stderr(), "{} frames", idx).unwrap();
        }

        println!("{}", 
            img.mask_iter(&mask).map(|x| x as u32).collect::<Vec<u32>>().mean()
        );
    }
}

