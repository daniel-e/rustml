//! Module with a collection of different mathematical functions.

extern crate num;

pub mod sum;
pub mod mean;
pub mod var;

pub use self::sum::Sum;
pub use self::mean::Mean;
pub use self::var::Var;

/// Determines the dimension over which to perform an operation.
pub enum Dimension {
    /// Perform the operation over all elements of a row.
    Row,
    /// Perform the operatino over all elements of a column.
    Column
}

/// Determines the type of normalization used for computing the variance
/// or standard deviation.
#[derive(Copy, Clone)]
pub enum Normalization {
    /// Use as denominator n, i.e. the number of examples.
    N,
    /// Use as denominator (n-1), i.e. the number of examples minus one.
    MinusOne
}

