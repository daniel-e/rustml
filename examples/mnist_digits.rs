extern crate rustml;

use std::process;
use rustml::*;
use rustml::datasets::MnistDigits;
use rustml::knn::classify;

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

fn main() {
    let k = 5;

    println!("Reading training data ...");
    let (training, training_labels) = get_data_sets(DataSet::Training);
    let tr = training.map(|&val| val as f32);

    println!("Reading test data ...");
    let (test, test_labels) = get_data_sets(DataSet::Test);
    let te = test.map(|&val| val as f32);

    // classify the first five examples from the test set
    println!("Classifying ...");
    let r = te.row_iter().zip(test_labels.iter()).take(5)
        .map(|(row, label)| {
            let target = classify(
                &tr, &training_labels, row, k, |x, y| Euclid::compute(x, y).unwrap()
            );

            (label, target)
        });

    for (x, y) in r {
        println!("label = {}, prediction = {}", x, y);
    }
}


