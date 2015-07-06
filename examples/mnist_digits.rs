extern crate rustml;

use rustml::*;
use rustml::knn::classify;
use rustml::datasets::MnistDigits;

fn main() {
    type T = f32;

    let k = 5;

    println!("Reading training data ...");
    let (training, training_labels) = 
        MnistDigits::from::<T>(
            "datasets/mnist_digits/train-images-idx3-ubyte.gz",
            "datasets/mnist_digits/train-labels-idx1-ubyte.gz"
        ).unwrap();

    println!("Reading test data ...");
    let (test, test_labels) = 
        MnistDigits::from::<T>(
            "datasets/mnist_digits/t10k-images-idx3-ubyte.gz",
            "datasets/mnist_digits/t10k-labels-idx1-ubyte.gz"
        ).unwrap();

    // classify the first five examples from the test set
    println!("Classifying ...");
    let r = test.row_iter().zip(test_labels.iter()).take(5)
        .map(|(row, label)| {
            let target = classify(
                &training, &training_labels, row, k, |x, y| Euclid::compute(x, y).unwrap()
            );

            (label, target)
        });

    for (x, y) in r {
        println!("label = {}, prediction = {}", x, y);
    }
}


