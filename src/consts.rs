use std::env::home_dir;
use std::path::Path;

// TODO os dependent path
pub static MNIST_PATH: &'static str = ".rustml/datasets/mnist_digits";

pub fn path_for(fname: &str) -> Result<String, &'static str> {

    match home_dir() {
        Some(ref mut p) => {
            p.push(Path::new(".rustml/datasets/"));
            p.push(Path::new(fname));
            println!("{}", p.as_path().to_str().unwrap().to_string());
            Ok(p.as_path().to_str().unwrap().to_string())
        }
        None => Err("Could not get home directory.")
    }
}

