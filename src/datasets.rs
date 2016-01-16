//! Module to easily access popular datasets often used to measure the performance of
//! machine learning algorithms.
//!
//! The image on the left shows an example of 200 handwritten digits from the 
//! [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/).
//! Rustml provides a simple interface via 
//! [MnistDigits](struct.MnistDigits.html) to easily access those digits. The MNIST
//! database comes with 60,000 examples in a training set and 10,000 examples in
//! a test set. The image has been created with rustml with just a few lines of code.
//! See the example [here](https://github.com/daniel-e/rustml/blob/master/examples/image_grid.rs).
//! 
//! The image in the middle shows an example of 1000 points normally distributed with the
//! mean at (1, 2) and a standard deviation of 0.3 for the first dimension and 0.4 for
//! the second dimension.
//!
//! The image on the right shows a mixture model. A dataset which consists of three sources
//! which are normally distributed with different parameters.
//!
//! <div style="font-size:80%">
//!  <div style="float:left;text-align:center;">
//!   MNIST database of handwritten digits<br/><img style="border-top:1px solid black" src="../../digits_grid.png">
//!  </div>
//!  <div style="float:left;text-align:center;padding-left:10px">
//!   Toy data: normally distributed data<br/><img src="../../plot_normal_1.png">
//!  </div>
//!  <div style="float:left;text-align:center;padding-left:10px">
//!   Toy data: mixture<br/><img src="../../plot_mixture.png">
//!  </div>
//! </div>
//! <div style="clear:both;"></div>

extern crate num;
extern crate time;
extern crate rand;

use std::io::Read;
use std::env::home_dir;
use std::path::Path;
use self::rand::distributions::{Normal, IndependentSample};
use self::rand::{SeedableRng, XorShiftRng};

use io::GzipData;
use matrix::*;

use consts::MNIST_PATH;

/// This structure offers access to the MNIST database of handwritten digits.
///
/// The database that is used is available at http://yann.lecun.com/exdb/mnist/
/// and contains 60,000 training examples and 10,000 test examples of handwritten
/// digits.
///
/// Currently, due to upload limits at crate.io a smaller training dataset
/// is used with 50,000 training examples.
///
/// To be able to use the dataset please follow the instructions mentioned
/// [here](https://github.com/daniel-e/rustml#datasets).
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

    fn read_examples(fname: &str) -> Result<Vec<u8>, &'static str> {

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

        Ok(v.to_vec())
    }

    pub fn from(vectors_fname: &str, labels_fname: &str) -> Result<(Matrix<u8>, Vec<u8>), &'static str> {

        let labels = try!(MnistDigits::read_labels(labels_fname));
        let values = try!(MnistDigits::read_examples(vectors_fname));

        match values.len() == labels.len() * 784 {
            true => {
                let m = Matrix::from_vec(values, labels.len(), 784);
                Ok((m, labels))
            },
            false => Err("Invalid matrix.")
        }
    }

    fn path(fname: &str) -> Result<String, &'static str> {

        match home_dir() {
            Some(ref mut p) => {
                p.push(Path::new(MNIST_PATH));
                p.push(Path::new(fname));
                Ok(p.as_path().to_str().unwrap().to_string())
            }
            None => Err("Could not get home directory.")
        }
    }

    /// Reads the default MNIST training set.
    ///
    /// Each row of the returned matrix represents an image of size 28x28.
    pub fn default_training_set() -> Result<(Matrix<u8>, Vec<u8>), &'static str> {

        // tested in tests directory
        let features = try!(MnistDigits::path("train-images-idx3-ubyte.gz"));
        let labels = try!(MnistDigits::path("train-labels-idx1-ubyte.gz"));
        MnistDigits::from(&features, &labels)
    }

    pub fn default_test_set() -> Result<(Matrix<u8>, Vec<u8>), &'static str> {

        // tested in tests directory
        let features = try!(MnistDigits::path("t10k-images-idx3-ubyte.gz"));
        let labels = try!(MnistDigits::path("t10k-labels-idx1-ubyte.gz"));
        MnistDigits::from(&features, &labels)
    }
}

// ----------------------------------------------------------------------------

/// Generates multi-dimensional data where each dimension is normally distributed.
///
/// # Example
///
/// ```
/// # extern crate rustml;
/// use rustml::datasets::*;
///
/// # fn main() {
/// let seed = [1, 2, 3, 4];
/// // create 2-dimensional vectors where the values of the first
/// // component are normal distributed with the parameters
/// // mu = 1.0 and std = 0.5 and the values of the second component
/// // are normal distributed with the parameters mu = 2.0 and
/// // std = 0.8.
/// let nd = 
///     normal_builder(seed)
///     .add(1.0, 0.5)  // parameters for the 1st dimension
///     .add(2.0, 0.8); // parameters for the 2nd dimension
/// for v in nd.take(100) {
///     // do s.th.
///     assert_eq!(v.len(), 2);
/// }
/// # }
/// ```
#[derive(Clone)]
pub struct NormalData {
    rng: XorShiftRng,
    normal: Vec<Normal>
}

impl NormalData {
    /// Adds a dimension for which the data is normally distributed with
    /// the given parameters.
    pub fn add(&self, mean: f64, std: f64) -> NormalData {

        let mut n = self.normal.clone();
        n.push(Normal::new(mean, std));

        NormalData {
            rng: self.rng.clone(),
            normal: n
        }
    }

    /// Returns the number of dimensions added via the `add` method.
    pub fn len(&self) -> usize { self.normal.len() }
}

impl Iterator for NormalData {
    type Item = Vec<f64>;
    
    fn next(&mut self) -> Option<Vec<f64>> {

        let n = self.normal.len();
        let mut v: Vec<f64> = vec![];

        for i in 0..n {
            v.push(self.normal[i].ind_sample(&mut self.rng));
        }
        Some(v)
    }
}

/// Creates a normally distributed data source.
pub fn normal_builder(seed: [u32; 4]) -> NormalData {
    NormalData { 
        rng: XorShiftRng::from_seed(seed),
        normal: vec![]
    }
}

// ----------------------------------------------------------------------------

/// Generates random multi-dimensional data points (a population) from
/// different normally distributed sources (subpopulations).
///
/// # Example
///
/// ```
/// # extern crate rustml;
/// use rustml::datasets::*;
///
/// # fn main() {
/// let seed = [2, 3, 5, 7];
/// let m = 
///     mixture_builder()
///         .add(100, normal_builder(seed).add(1.0, 1.2).add(2.0, 1.2))
///         .add(100, normal_builder(seed).add(5.0, 1.5).add(6.0, 1.5))
///         .add(100, normal_builder(seed).add(6.0, 1.5).add(0.0, 1.5))
///         .as_matrix();
/// assert_eq!(m.rows(), 300);
/// assert_eq!(m.cols(), 3);
/// # }
/// ```
pub struct Mixture {
    normal: Vec<(usize, NormalData)>
}

impl Mixture {

    /// Adds a normally distributed data source (subpopulation) 
    /// which generates `n` samples.
    pub fn add(&self, n: usize, src: NormalData) -> Mixture {

        if self.normal.len() > 0 && self.normal[0].1.len() != src.len() {
            panic!("Invalid length.");
        }

        let mut v = self.normal.clone();
        v.push((n, src));
        Mixture {
            normal: v
        }
    }

    /// Returns a matrix which contains the population consisting of one or
    /// more subpopulations.
    ///
    /// Each row of the matrix represents a sample generated from one of
    /// the subpopulations. 
    ///
    /// The value in the first column of
    /// a row denotes the subpopulation from which this sample has been generated.
    /// If the value is 0 this sample has been
    /// created from the first subpopulation (i.e. the first data source that has
    /// been added with the `add` method). If the value is 1 this sample has been
    /// created from the second subpopulation and so on.
    ///
    /// The following columns denote the dimensions of the data sources.
    ///
    /// # Example
    ///
    /// Let's assume you create a mixture model with two data sources
    /// each with two dimensions as follows:
    ///
    /// ```ignore
    /// let m = 
    ///     mixture_builder()
    ///         .add(100, normal_builder(seed).add(1.0, 0.2).add(2.0, 0.2))
    ///         .add(100, normal_builder(seed).add(5.0, 0.2).add(6.0, 0.2))
    ///         .as_matrix();
    /// ```
    ///
    /// Then, the matrix could look like:
    /// 
    pub fn as_matrix(&mut self) -> Matrix<f64> {

        let mut m = Matrix::new();

        for (idx, &mut (n, ref mut nd)) in self.normal.iter_mut().enumerate() {
            for _ in 0..n {
                let v = nd.next().unwrap().clone();
                let mut x = vec![idx as f64];
                for j in v {
                    x.push(j);
                }
                m.add_row(&x);
            }
        }
        m
    }
}

/// Creates a mixture model with normally distributed data sources.
pub fn mixture_builder() -> Mixture {
    Mixture {
        normal: vec![]
    }
}

// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use io::GzipData;

    #[test]
    fn test_from_high_endian() {

        let a = [1, 2, 3];
        assert_eq!(
            MnistDigits::from_high_endian(&a), 
            (1 * 256 + 2) * 256 + 3
        );

        let b = [255, 254, 253, 252];
        assert_eq!(
            MnistDigits::from_high_endian(&b), 
            ((255 * 256 + 254) * 256 + 253) * 256 + 252
        );
    }

    #[test]
    fn test_read_u32() {

        let mut gz = GzipData::from_buf(vec![1, 2, 3]);
        assert!(MnistDigits::read_u32(&mut gz).is_err());

        gz = GzipData::from_buf(vec![1, 2, 3, 4]);
        assert_eq!(
            MnistDigits::read_u32(&mut gz).unwrap(), 
            ((256 * 1 + 2) * 256 + 3) * 256 + 4
        );
    }

    #[test]
    fn test_normal_data() {
        let n = normal_builder([1,2,3,4]).add(0.0, 2.0).add(1.0, 1.0);

        let data = n.take(5).collect::<Vec<Vec<f64>>>();
        assert_eq!(data.len(), 5);
        assert_eq!(data[0].len(), 2);
    }

    #[test]
    fn test_mixture() {

        let seed = [1, 2, 3, 4];
        let m = mixture_builder()
            .add(3, normal_builder(seed).add(1.0, 0.5).add(2.0, 1.0))
            .add(5, normal_builder(seed).add(3.0, 0.5).add(4.0, 1.0))
            .add(8, normal_builder(seed).add(2.0, 0.5).add(7.0, 1.0))
            .as_matrix();
        assert_eq!(m.rows(), 16);
        assert_eq!(m.cols(), 3);
        // TODO more tests
    }

}
