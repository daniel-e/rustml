extern crate rustml;

use std::fs::File;
use std::io::Write;
use std::process;
use rustml::*;
use rustml::datasets::MnistDigits;

enum DataSet {
    Training,
    Test,
}

fn get_data_sets(d: DataSet) -> (Matrix<u8>, Vec<u8>) {

    let r = match d {
        DataSet::Training => MnistDigits::default_training_set(),
        DataSet::Test => MnistDigits::default_test_set(),
    };

    match r {
        Err(_) => {
            println!("\n\nThis example requires the MNIST dataset of handwritten digits.");
            println!("Please have a look at https://github.com/daniel-e/rustml#rustml-datasets-package");
            println!("for details on how to install it.\n");
            process::exit(0);
        },
        Ok(data) => data
    }
}

fn write_matrix(m: Matrix<f32>, fname: &str) {

    println!("Writing {}", fname);

    let mut f = File::create(fname).unwrap();
    for row in m.row_iter() {
        f.write_all(row.to_vec().to_string().as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

fn write_vec(v: Vec<u8>, fname: &str) {

    println!("Writing {}", fname);

    let mut f = File::create(fname).unwrap();
    for val in v.iter() {
        let s = format!("{}\n", val);
        f.write_all(s.as_bytes()).unwrap();
    }
}

fn main() {

    println!("Reading training data ...");
    let (training, training_labels) = get_data_sets(DataSet::Training);
    let tr = training.map(|&val| val as f32 / 255.0);

    write_matrix(tr, "/tmp/mnist_training.txt");
    write_vec(training_labels, "/tmp/mnist_training_labels.txt");

    println!("Reading test data ...");
    let (test, test_labels) = get_data_sets(DataSet::Test);
    let te = test.map(|&val| val as f32 / 255.0);

    write_matrix(te, "/tmp/mnist_test.txt");
    write_vec(test_labels, "/tmp/mnist_test_labels.txt");

    // load the data in octave
    // trainX = load("mnist_training.txt");
    // trainY = load("mnist_training_labels.txt");
    // testX = load("mnist_test.txt");
    // testY = load("mnist_test_labels.txt");
    //
    // write all variables compressed into one file
    // save -z "mnist.txt.zip" trainX trainY testX testY
}


