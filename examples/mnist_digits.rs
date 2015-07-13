extern crate rustml;

use rustml::*;
use rustml::datasets::MnistDigits;
use rustml::knn::classify;

fn main() {
    let k = 5;

    println!("Reading training data ...");
    let (training, training_labels) = MnistDigits::default_training_set().unwrap();
    let tr = training.map(|&val| val as f32);

    println!("Reading test data ...");
    let (test, test_labels) = MnistDigits::default_test_set().unwrap();
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


