
//use std::fs::File;

use std::io::Read;
use ::io::GzipData;
use ::matrix::*;

pub struct MnistDigits;

pub fn from_high_endian(arr: &[u8]) -> u64 {

    let mut val: u64 = 0;
    for i in arr.iter() {
        val = val * 256 + *i as u64;
    }
    val
}

impl MnistDigits {

    // format of mnist data set
    // http://yann.lecun.com/exdb/mnist/

    pub fn training_set() -> Result<Matrix<f64>, &'static str> {

        // TODO location of dataset
        let mut labels = try!(GzipData::from_file("datasets/mnist_digits/train-labels-idx1-ubyte.gz"));

        let mut magic: [u8; 4] = [0; 4];
        try!(labels.read(&mut magic).map_err(|_| "Could not read magic number."));
        if magic != [0, 0, 8, 1] {
            return Err("Invalid magic number.");
        }

        let mut nitems: [u8; 4] = [0; 4];
        try!(labels.read(&mut nitems).map_err(|_| "Could not read number of items."));
        if from_high_endian(&nitems) != 60000 {
            return Err("Invalid number of items.");
        }

        // TODO

        Ok(Matrix::<f64>::new())
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_set() {

        assert!(MnistDigits::training_set().is_ok());
    }

    #[test]
    fn test_from_high_endian() {

        let a = [1, 2, 3];
        assert_eq!(from_high_endian(&a), (1 * 256 + 2) * 256 + 3);
    }

}
