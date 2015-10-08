//! Library for doing machine learning with Rust.
//!
//! <div style="font-size:80%">
//!  <div style="float:left;text-align:center">
//!   Linear regression<br/><img src="../linreg_plot.png">
//!  </div>
//!  <div style="float:left;text-align:center;padding-left:10px">
//!   MNIST database of handwritten digits<br/><img style="border-top:1px solid black" src="../digits_grid.png">
//!  </div>
//!  <div style="float:left;text-align:center;padding-left:10px">
//!   Gradient descent<br/><img style="border-top:1px solid black" src="../gradient_descent.png">
//!  </div>
//!
//!  <div style="clear:both;"></div>
//!
//!  <!-- <div style="float:left;text-align:center">
//!   Generator for normally distributed data<br/><img src="../plot_normal_1.png">
//!  </div> -->
//!  <div style="float:left;text-align:center">
//!   Neural networks<br/><img src="../nn.png">
//!  </div>
//!  <div style="float:left;text-align:center">
//!   Toy data: mixture<br/><img src="../plot_mixture.png">
//!  </div>
//!  <div style="float:left;text-align:center">
//!   Decision boundary for knn (k = 5)<br/><img src="../plot_knn_boundary.png">
//!  </div>
//!
//!  <div style="clear:both;"></div>
//! </div>
//!
//! # Features
//! <i>(click on a link to get more details)</i>
//!
//! * [highly optimized linear algebra via BLAS integration](blas/index.html) (i.e. operations on vectors and
//! matrices)
//! * gradient descent with debugging capabilities (e.g. with learning curves)
//! * [neural networks](nn/index.html)
//! * DBSCAN clustering algorithm
//! * linear regression
//! * optimization of linear regression with gradient descent
//! * classification with <i>k</i>-nearest neighbours
//! * sliding windows for arbitrary dimensions (e.g. for image processing)
//! * [standard databases](datasets/index.html) (e.g. MNIST database of handwritten digits)
//! * feature scaling
//! * video and image processing via integration of OpenCV
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
//!
//! # Machine Learning Pipelines with Rustml
//!
//! The Rustml pipeline is a small and simple framework to build
//! and configure machine learning pipelines that have been shown to be a quite
//! powerful technique when doing machine learning. How pipelines can be
//! created with Rustml can be seen
//! [here](https://github.com/daniel-e/rustml/tree/master/pipeline).
//!
//! # Example how to do classifications
//! 
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
//! neighbor](https://github.com/daniel-e/rustml/blob/master/examples/mnist_digits.rs): classifies the examples of the test set of the MNIST database of handwritten digits with a simple <i>k</i>-nearest neighbor approach. (this examples requires an external dataset that has to be downloaded separately (see [here](https://github.com/daniel-e/rustml#datasets))
//! * [matrix
//! multiplication](https://github.com/daniel-e/rustml/blob/master/examples/matrix_multiplication.rs): multiplies two 6000x6000 matrices.
//! * [vector
//! addition](https://github.com/daniel-e/rustml/blob/master/examples/vector_addition.rs): add
//! vectors
//! * [feature scaling](https://github.com/daniel-e/rustml/blob/master/examples/scale_matrix.rs)
//! * [feature extraction from all frames of a
//! video](https://github.com/daniel-e/rustml/blob/master/examples/video_histogram.rs): this
//! examples requires an external dataset that has to be downloaded separately (see
//! [here](https://github.com/daniel-e/rustml#datasets)).
//! * [plot with octave](https://github.com/daniel-e/rustml/blob/master/examples/octave_plot.rs):
//! create plots with Octave
//! * [gradient descent](https://github.com/daniel-e/rustml/blob/master/examples/gradient_descent.rs): use gradient
//! to optimize a function with two parameters
//! * [image grid](https://github.com/daniel-e/rustml/blob/master/examples/image_grid.rs): plot some of the 
//! handwritten digits of the MNIST database into a grid
//!
pub use distance::{Distance, Euclid, DistancePoint2D};
pub use matrix::{HasNan, Similar, Matrix};
pub use math::{Dimension, Normalization, Mean, MeanVec, Sum, Var, SumVec};
pub use ops::{MatrixScalarOps, Ops, VectorScalarOps, VectorVectorOps, MatrixMatrixOps};
pub use ops_inplace::{VectorVectorOpsInPlace, MatrixMatrixOpsInPlace};
pub use gaussian::{GaussianEstimator, GaussianFunctions, Gaussian};
pub use geometry::{Point2D};
pub use vectors::{Linspace, VectorIO};
pub use datasets::{mixture_builder, normal_builder};

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
pub mod scaling;
pub mod opencv;
pub mod geometry;
pub mod dbscan;
pub mod sliding;
pub mod hash;
pub mod opt;
pub mod octave;
pub mod regression;
pub mod nn;
