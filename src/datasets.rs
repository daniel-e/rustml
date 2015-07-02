extern crate num;
extern crate std;
extern crate time;

use std::io::Read;
use self::num::traits::Float;

use io::GzipData;
use matrix::*;

/// This structure offers access to the MNIST database of handwritten digits.
///
/// The database that is used is available at http://yann.lecun.com/exdb/mnist/
/// and contains 60,000 training examples and 10,000 test examples of handwritten
/// digits.
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

    fn read_labels(fname: &str, n: u32) -> Result<Vec<u8>, &'static str> {

        let mut data = try!(GzipData::from_file(fname));

        if try!(MnistDigits::read_u32(&mut data)) != 8 * 256 + 1 {
            return Err("Invalid magic number.");
        }

        let k = try!(MnistDigits::read_u32(&mut data));
        if k != n {
            return Err("Invalid number of items.");
        }

        let l: Vec<u8> = data.iter().cloned().collect();
        if l.len() != n as usize {
            return Err("Invalid number of items.");
        }

        if !l.iter().all(|&x| x <= 9) {
            return Err("Found invalid values for labels.");
        }
        
        Ok(l)
    }

    fn read_examples<T: Float + Cast>(fname: &str, n: u32) -> Result<Vec<T>, &'static str> {

        let mut data = try!(GzipData::from_file(fname));

        if try!(MnistDigits::read_u32(&mut data)) != 8 * 256 + 3 {
            return Err("Invalid magic number.");
        }

        if try!(MnistDigits::read_u32(&mut data)) != n {
            return Err("Invalid number of items.");
        }

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
            r.push(T::cast(*i));
        }
        Ok(r)
    }

    pub fn training_set<T: Float + Cast>() -> Result<(Matrix<T>, Vec<u8>), &'static str> {

        // TODO location of dataset
        let labels = try!(MnistDigits::read_labels("datasets/mnist_digits/train-labels-idx1-ubyte.gz", 60000));
        let values = try!(MnistDigits::read_examples::<T>("datasets/mnist_digits/train-images-idx3-ubyte.gz", 60000));

        let m = Matrix::from_vec(values, 60000, 784);
        match m {
            Some(mat) => Ok((mat, labels)),
            _ => Err("Could not create matrix.")
        }
    }

    pub fn test_set<T: Float + Cast>() -> Result<(Matrix<T>, Vec<u8>), &'static str> {
        // TODO location of dataset
        let labels = try!(MnistDigits::read_labels("datasets/mnist_digits/t10k-labels-idx1-ubyte.gz", 10000));
        let values = try!(MnistDigits::read_examples::<T>("datasets/mnist_digits/t10k-images-idx3-ubyte.gz", 10000));

        let m = Matrix::from_vec(values, 10000, 784);
        match m {
            Some(mat) => Ok((mat, labels)),
            _ => Err("Could not create matrix.")
        }
    }
}

pub trait Cast {
    fn cast(x: u8) -> Self;
}

impl Cast for f32 {
    fn cast(x: u8) -> f32 { x as f32 }
}

impl Cast for f64 {
    fn cast(x: u8) -> f64 { x as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::io::GzipData;
    use std::io::Read;

    #[test]
    fn test_training_set() {


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
    fn test_performance() {

        let mut gz = GzipData::from_file("datasets/mnist_digits/train-images-idx3-ubyte.gz").unwrap();
        let mut v: Vec<u8> = Vec::new();
        assert_eq!(gz.read_to_end(&mut v).unwrap(), 47040016);
    }
}
