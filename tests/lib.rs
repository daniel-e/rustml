#[macro_use] extern crate rustml;

use std::env::home_dir;
use std::path::Path;
use std::io::Read;

use rustml::io::GzipData;
use rustml::datasets::MnistDigits;
use rustml::consts::MNIST_PATH;

#[test]
fn datasets_mnist() {

    let (training, training_labels) = MnistDigits::default_training_set().unwrap();
    assert_eq!(training.rows(), 60000);
    assert_eq!(training.cols(), 28 * 28);
    assert_eq!(training_labels.len(), 60000);

    let (testing, testing_labels) = MnistDigits::default_test_set().unwrap();
    assert_eq!(testing.rows(), 10000);
    assert_eq!(testing.cols(), 28 * 28);
    assert_eq!(testing_labels.len(), 10000);
}

#[test]
fn test_read_mnist_training_set() {

    let mut p = home_dir().unwrap();
    p.push(Path::new(MNIST_PATH));
    p.push(Path::new("train-images-idx3-ubyte.gz"));

    let mut gz = GzipData::from_file(p.as_path().to_str().unwrap()).unwrap();
    let mut v: Vec<u8> = Vec::new();
    assert_eq!(gz.read_to_end(&mut v).unwrap(), 47040016);
}

