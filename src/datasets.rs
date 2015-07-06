//! Module to easily access popular datasets often used to measure the performance of
//! machine learning algorithms.

extern crate num;
extern crate std;
extern crate time;

use std::io::Read;
use self::num::traits::{Float, FromPrimitive};

use io::GzipData;
use matrix::*;

/// This structure offers access to the MNIST database of handwritten digits.
///
/// The database that is used is available at http://yann.lecun.com/exdb/mnist/
/// and contains 60,000 training examples and 10,000 test examples of handwritten
/// digits.
///
/// Currently, due to upload limits at crate.io a smaller training dataset
/// is used with 50,000 training examples.
pub struct MnistDigits;

impl MnistDigits {

    fn from_high_endian(arr: &[u8]) -> u64 {

        arr.iter().fold(0, |acc, val| acc * 256 + (*val as u64))
    }

    fn read_u32<T: Read>(src: &mut T) -> Result<u32, &'static str> {

        let mut buf: [u8; 4] = [0; 4];

        match src.read(&mut buf) {
            Ok(4) => Ok(MnistDigits::from_high_endian(&buf) as u32),
            _     => Err("Could not read data."),
        }
     }

    fn read_labels(fname: &str) -> Result<Vec<u8>, &'static str> {

        let mut data = try!(GzipData::from_file(fname));

        if try!(MnistDigits::read_u32(&mut data)) != 8 * 256 + 1 {
            return Err("Invalid magic number.");
        }

        let n = try!(MnistDigits::read_u32(&mut data));

        let l: Vec<u8> = data.iter().cloned().collect();
        if l.len() != n as usize {
            return Err("Invalid number of items.");
        }

        if !l.iter().all(|&x| x <= 9) {
            return Err("Found invalid values for labels.");
        }
        
        Ok(l)
    }

    fn read_examples<T: Float + FromPrimitive>(fname: &str) -> Result<Vec<T>, &'static str> {

        let mut data = try!(GzipData::from_file(fname));

        if try!(MnistDigits::read_u32(&mut data)) != 8 * 256 + 3 {
            return Err("Invalid magic number.");
        }

        let n = try!(MnistDigits::read_u32(&mut data));

        let rows = try!(MnistDigits::read_u32(&mut data));
        let cols = try!(MnistDigits::read_u32(&mut data));
        if rows != 28 || cols != 28 {
            return Err("Invalid number of rows or columns.");
        }

        let v = data.buf();

        if v.len() != (n * 28 * 28) as usize {
            return Err("Could not read data.");
        }

        let mut r: Vec<T> = Vec::with_capacity(v.len());
        for i in v {
            r.push(T::from_u8(*i).unwrap());
        }
        Ok(r)
    }

    pub fn from<T: Float + FromPrimitive>(vectors_fname: &str, labels_fname: &str) -> Result<(Matrix<T>, Vec<u8>), &'static str> {

        let labels = try!(MnistDigits::read_labels(labels_fname));
        let values = try!(MnistDigits::read_examples::<T>(vectors_fname));

        match Matrix::from_vec(values, labels.len(), 784) {
            Some(matrix) => {
                if matrix.rows() != labels.len() {
                    return Err("Number of examples are different.");
                }
                Ok((matrix, labels))
            }
            _ => Err("Invalid matrix.")
        }
    }

    pub fn training_set<T: Float + FromPrimitive>() -> Result<(Matrix<T>, Vec<u8>), &'static str> {

        // TODO location of dataset
        MnistDigits::from(
            "datasets/mnist_digits/train-images-idx3-ubyte.50k.gz",
            "datasets/mnist_digits/train-labels-idx1-ubyte.50k.gz"
        )
    }

    pub fn test_set<T: Float + FromPrimitive>() -> Result<(Matrix<T>, Vec<u8>), &'static str> {

        // TODO location of dataset
        MnistDigits::from(
            "datasets/mnist_digits/t10k-images-idx3-ubyte.gz",
            "datasets/mnist_digits/t10k-labels-idx1-ubyte.gz"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::io::GzipData;
    //use std::io::Read;

    #[test]
    fn test_training_set() {

        // TODO
        //let (training, training_labels) = MnistDigits::training_set().unwrap();
        //let (testing, testing_labels) = MnistDigits::test_set().unwrap();
    }

    #[test]
    fn test_from_high_endian() {

        let a = [1, 2, 3];
        assert_eq!(MnistDigits::from_high_endian(&a), (1 * 256 + 2) * 256 + 3);
        let b = [255, 254, 253, 252];
        assert_eq!(MnistDigits::from_high_endian(&b), ((255 * 256 + 254) * 256 + 253) * 256 + 252);
    }

    #[test]
    fn test_read_u32() {

        let mut gz = GzipData::from_buf(vec![1, 2, 3]);
        assert!(MnistDigits::read_u32(&mut gz).is_err());

        gz = GzipData::from_buf(vec![1, 2, 3, 4]);
        assert_eq!(MnistDigits::read_u32(&mut gz).unwrap(), ((256 * 1 + 2) * 256 + 3) * 256 + 4);
    }

    #[test]
    fn test_read() {
        // TODO
        /*
        let mut gz = GzipData::from_file("datasets/mnist_digits/train-images-idx3-ubyte.gz").unwrap();
        let mut v: Vec<u8> = Vec::new();
        assert_eq!(gz.read_to_end(&mut v).unwrap(), 47040016);
        */
    }
}
