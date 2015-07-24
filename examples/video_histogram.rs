extern crate rustml;

use std::io::Write;
use rustml::*;
use rustml::opencv::*;

fn main() {
    
    let video = Video::from_file("/home/dz/videos/generated/reduced.avi").unwrap();
    let mask = GrayImage::from_file("/home/dz/videos/generated/mask.png").unwrap();

    let mut values: Vec<u32> = Vec::new();
    let mut c = 0;

    for i in video.gray_frame_iter() {
        values.push(
            i.pixels_from_mask_as_u8(&mask).unwrap()
             .map(|&x| x as u32)
             .mean()
        );

        c += 1;
        if c % 1000 == 0 {
            writeln!(&mut std::io::stderr(), "{}", c);
        }

        println!("{}", values.last().unwrap());
    }
}

