//! Library for doing machine learning with Rust.
//!
//! # Performance
//!
//! When it comes to vector and matrix operations rustml makes heavy
//! use of highly optimized numeric libraries like [BLAS](http://www.netlib.org/blas/)
//! or [ATLAS](http://math-atlas.sourceforge.net/). By default CBLAS is used because
//! it is installed on many systems by default. However, in many cases performance
//! can be greatly improved when switching to ATLAS. For a detailed description on how
//! to optimize the numeric computations please read the separate
//! documentation on this topic available
//! [here](https://github.com/daniel-e/rustml/tree/master/build).
//!
//! # Example how to do classifications
//! In the following example a simple k-nearest neighbour algorithm is used to predict
//! the label of a vector with two features based on the examples in the matrix
//! `m` (the training set) with their known labels stored in the vector `labels`.
//!
//! ```
//! # #[macro_use] extern crate rustml;
//! use rustml::*;
//!
//! # fn main() {
//! let m = mat![  // training set
//!     1.0, 2.0;  // each row contains one example for which the label is
//!     1.1, 2.1;  // known
//!     2.0, 3.0;
//!     0.9, 1.9;
//!     2.1, 2.9
//! ];
//!
//! let labels = vec![1, 2, 2, 1, 2];
//!
//! // predict the label for feature vector [1.3, 2.0]
//! let target = 
//!     knn::classify(
//!         &m, &labels, &[1.3, 2.0], 
//!         3, // look at the 3 nearest neighbours to make the decision
//!         |x, y| Euclid::compute(x, y).unwrap() // use Euclidean distance
//!     );
//! assert_eq!(target, 1);
//! # }
//! ```
//!
//! # All examples
//!
//! * [<i>k</i>-nearest
//! neighbor](https://github.com/daniel-e/rustml/blob/master/examples/mnist_digits.rs): Classifies the examples of the test set of the MNIST database of handwritten digits with a simple <i>k</i>-nearest neighbor approach.
//! * [matrix
//! multiplication](https://github.com/daniel-e/rustml/blob/master/examples/matrix_multiplication.rs): Multiplies two 6000x6000 matrices.
//! * [vector
//! addition](https://github.com/daniel-e/rustml/blob/master/examples/vector_addition.rs):
//! Add vectors.

pub use distance::{Distance, Euclid};
pub use matrix::{HasNan, Matrix};
pub use math::{Dimension, Normalization, Mean, Sum, Var};
pub use ops::{MatrixScalarOps, VectorScalarOps};
pub use ops_inplace::{VectorVectorOpsInPlace};
pub use gaussian::{GaussianEstimator, GaussianFunctions, Gaussian};

// ordering is important because the macro mat! is 
// only available for modules which follow #[macro_use]
#[macro_use]
pub mod matrix;

pub mod blas;
pub mod csv;
pub mod datasets;
pub mod distance;
pub mod io;
pub mod knn;
pub mod norm;
pub mod vectors;
pub mod math;
pub mod gaussian;
pub mod ops;
pub mod consts;
pub mod ops_inplace;
