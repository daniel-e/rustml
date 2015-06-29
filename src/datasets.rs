
//use std::fs::File;

use std::io::Read;
use ::io::GzipData;
use ::matrix::*;

/// This structure offers access to the MNIST database of handwritten digits.
///
/// The database that is used is available at http://yann.lecun.com/exdb/mnist/
/// and contains 60,000 training examples and 10,000 test examples of handwritten
/// digits.
pub struct MnistDigits;

impl MnistDigits {

    fn from_high_endian(arr: &[u8]) -> u64 {

        let mut val: u64 = 0;
        for i in arr.iter() {
            val = val * 256 + *i as u64;
        }
        val
    }

    fn read_u32<T: Read>(src: &mut T) -> Result<u32, &'static str> {

        let mut buf: [u8; 4] = [0; 4];
        try!(src.read(&mut buf).map_err(|_| "Could not read data."));
        Ok(MnistDigits::from_high_endian(&buf) as u32)
     }

    pub fn training_set() -> Result<Matrix<f64>, &'static str> {

        // TODO location of dataset
        let mut labels = try!(GzipData::from_file("datasets/mnist_digits/train-labels-idx1-ubyte.gz"));

        let magic = try!(MnistDigits::read_u32(&mut labels));
        if magic != 8 * 256 + 1 {
            return Err("Invalid magic number.");
        }

        let nitems = try!(MnistDigits::read_u32(&mut labels));
        if nitems != 60000 {
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
        assert_eq!(MnistDigits::from_high_endian(&a), (1 * 256 + 2) * 256 + 3);
    }

}
