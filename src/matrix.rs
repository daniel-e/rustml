//! Module that contains structs and functions useful for doing matrix
//! operations.

extern crate libc;
extern crate rand;
extern crate num;

use std::{iter, fmt, fs};
use std::iter::FromIterator;
use std::io::Read;
use std::str::FromStr;
use std::ops::Mul;
use std::slice::{Iter, IterMut};
use self::rand::{thread_rng, Rng, Rand};
use self::num::traits::{Float, Signed};
use self::libc::{c_int, c_double, c_float};

use blas::{Order, Transpose, cblas_dgemm, cblas_sgemm};

// TODO implement some ops
// https://doc.rust-lang.org/std/ops/


// ------------------------------------------------------------------

/// A matrix with elements of type T.
///
/// An `Option` is returned which is `None` if the matrices cannot be 
/// multiplied because the number of columns of the left matrix is not
/// equal to the number of rows of the right matrix.
///
/// Because the trait 
/// [Mul](http://doc.rust-lang.org/nightly/core/ops/trait.Mul.html) is 
/// implemented for matrices where `T` is `f32` or `f64` such matrices
/// can be multplied with the `*` operator.
///
/// # Creating a matrix
///
/// ## FromIterator
///
/// The trait `FromIterator` is implemented so that a matrix can be created
/// with the code below. The `collect` function can only create matrices
/// with one row. You can use the `reshape` or `reshape_mut` function to
/// reshape the matrix.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// # use rustml::*;
///
/// # fn main() {
/// let a = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let m = a.iter().cloned()
///     .collect::<Matrix<_>>() // collect into a matrix with 1 row and 8 columns
///     .reshape(2, 4);         // reshape into a 2x4 matrix
/// assert_eq!(m, mat![1, 2, 3, 4; 5, 6, 7, 8]);
/// # }
/// ```
///
/// ## IntoMatrix
///
/// A matrix can be easily created from each data type which implements the
/// `IntoMatrix` trait of the `matrix` module.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// # use rustml::*;
///
/// # fn main() {
/// let a = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// // create a matrix from a with 4 rows and 2 columns
/// let m = a.to_matrix(4);
/// assert_eq!(m, mat![1, 2; 3, 4; 5, 6; 7, 8]);
/// # }
/// ```
/// # Multiplication of matrices
///
/// Two matrices can be easily multiplied by using the `*` operator. The
/// multiplication is implemented for `f32` and `f64` and it uses BLAS to
/// do this in a very efficient manner. For a detailed description on how
/// to optimize the numeric computations please read the separate
/// documentation on this topic available
/// [here](https://github.com/daniel-e/rustml/tree/master/build).
///
/// ## Example: matrix multiplication
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::*;
///
/// # fn main() {
/// let a = mat![
///     1.0f32, 5.0, 2.0; 
///     2.0, 2.0, 3.0 
/// ];
/// let b = mat![
///     3.0, 7.0, 4.0, 8.0;
///     4.0, 2.0, 1.0, 4.0;
///     5.0, 2.0, 1.0, 9.0
/// ];
/// let c = (a * b).unwrap();
/// assert_eq!(c.row(0).unwrap(), &[33.0, 21.0, 11.0, 46.0]);
/// assert_eq!(c.row(1).unwrap(), &[29.0, 24.0, 13.0, 51.0]);
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<T> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>
}

// ------------------------------------------------------------------

/// Trait to convert a data type into a matrix.
pub trait IntoMatrix<T> {

    /// Converts a data type into a matrix.
    ///
    /// The number of rows is specified by the parameter
    /// `rows`.
    ///
    /// Panics if the number of elements from which the matrix
    /// is created is not divisible by `rows`.
    fn to_matrix(&self, rows: usize) -> Matrix<T>;
}

impl <T: Clone> IntoMatrix<T> for Vec<T> {

    fn to_matrix(&self, rows: usize) -> Matrix<T> {
        assert!(self.len() % rows == 0, 
            "The length of the vector must be divisible by the number of rows."
        );
        Matrix::from_vec(self.clone(), rows, self.len() / rows).unwrap()
    }
}

impl <T: Clone> IntoMatrix<T> for [T] {

    fn to_matrix(&self, rows: usize) -> Matrix<T> {
        assert!(self.len() % rows == 0, 
            "The length of the vector must be divisible by the number of rows."
        );
        Matrix::from_vec(self.to_vec(), rows, self.len() / rows).unwrap()
    }
}

// ------------------------------------------------------------------

/// Trait to check if a matrix contains a NaN value.
pub trait HasNan {
    /// Returns `true` if at least one element that is NaN exists.
    fn has_nan(&self) -> bool;
}

impl <T: Float> HasNan for Matrix<T> {

    /// Returns `true` if the matrix contains at least one element that is NaN.
    fn has_nan(&self) -> bool {
        self.data.iter().any(|&x| x.is_nan())
    }
}

// ------------------------------------------------------------------

/// Trait to check if the values of two matrices with the same dimension
/// differ only within a small range.
pub trait Similar<T> {
    fn similar(&self, e: &Self, epsilon: T) -> bool;
}

impl <T: Clone + Signed + Float> Similar<T> for Matrix<T> {

    fn similar(&self, e: &Self, epsilon: T) -> bool {

        assert!(
            self.rows() == e.rows() && 
            self.cols() == e.cols(), 
            format!("Dimensions of matrices do not match. {}x{}, {}x{}",
                    self.rows(), self.cols(), e.rows(), e.cols()
            )
        );

        self.values().zip(e.values()).all(|(&x, &y)| num::abs(x - y) <= epsilon)
    }
}

impl <T: Clone + Signed + Float> Similar<T> for Vec<T> {

    fn similar(&self, e: &Self, epsilon: T) -> bool {

        self[..].similar(e, epsilon)
    }
}

impl <T: Clone + Signed + Float> Similar<T> for [T] {

    fn similar(&self, e: &Self, epsilon: T) -> bool {

        assert!(self.len() == e.len(), "Dimensions of vectors do not match.");
        self.iter().zip(e.iter()).all(|(&x, &y)| num::abs(x - y) <= epsilon)
    }
}

// --------------- Matrix macro mat! --------------------------------

/// Macro to create a matrix.
///
/// # Example
///
/// let m = mat![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
///
/// This example creates the following 2x3 matrix:
///
/// `[ 1 2 3 ]`
///
/// `[ 4 5 6 ]`
#[macro_export]
macro_rules! mat {
    ( $( $( $x:expr ),+ ) ;* ) => {
        {
        let mut v = Vec::new();
        let mut cols;
        let mut cols_old = 0;
        let mut rows = 0;
        $(
            rows += 1;
            cols = 0;
            $(
                cols += 1;
                v.push($x);
            )+
            if rows > 1 && cols != cols_old {
                panic!("Invalid matrix.");
            }
            cols_old = cols;
        )*
        Matrix::from_vec(v, rows, cols_old).unwrap()
        }
    };
}

// ------------------------------------------------------------------

impl <T: Clone> FromIterator<T> for Matrix<T> {

    fn from_iter<I>(iterator: I) -> Self
        where I: IntoIterator<Item = T> {

        let v = Vec::from_iter(iterator);
        let n = v.len();
        Matrix::<T>::from_vec(v, 1, n).unwrap()
    }
}

// ------------------------------------------------------------------

impl <T: Clone> Matrix<T> {

    // Functions for constructing a matrix.

    /// Creates a new matrix with 0 rows and 0 columns.
    ///
    /// ```
    /// use rustml::Matrix;
    ///
    /// let m = Matrix::<f32>::new();
    /// assert_eq!(m.rows(), 0);
    /// assert_eq!(m.cols(), 0);
    /// ```
    pub fn new() -> Matrix<T> {

        Matrix::from_vec(Vec::new(), 0, 0).unwrap()
    }

    /// Creates a matrix with the given number of rows and columns
    /// where each element is set to `value`.
    ///
    /// ```
    /// use rustml::Matrix;
    ///
    /// let m = Matrix::<f32>::fill(1.2, 2, 2);
    /// assert_eq!(m.row(0).unwrap(), [1.2, 1.2]);
    /// assert_eq!(m.row(1).unwrap(), [1.2, 1.2]);
    /// ```
    pub fn fill(value: T, rows: usize, cols: usize) -> Matrix<T> {

        Matrix::from_vec(
            iter::repeat(value).take(rows * cols).collect(),
            rows, cols
        ).unwrap()
    }

    /// Creates a matrix with the given number of rows and columns. The matrix is
    /// initialized with the values from the vector `vals`. The elements
    /// are arranged in row-major order in the vector.
    ///
    /// ```
    /// use rustml::Matrix;
    ///
    /// let m = Matrix::<f32>::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2
    /// ).unwrap();
    /// assert_eq!(m.row(0).unwrap(), [1.0, 2.0]);
    /// assert_eq!(m.row(1).unwrap(), [3.0, 4.0]);
    /// assert_eq!(m.row(2).unwrap(), [5.0, 6.0]);
    /// ```
    pub fn from_vec(vals: Vec<T>, rows: usize, cols: usize) -> Option<Matrix<T>> {

        match rows * cols == vals.len() {
            false => None,
            true  => Some(Matrix {
                nrows: rows,
                ncols: cols,
                data: vals
            })
        }
    }

    /// Creates a matrix from a vector of column vectors.
    pub fn from_col_vectors(v: &[Vec<T>]) -> Option<Matrix<T>> {

        if v.len() == 0 {
            return Some(Matrix::new());
        }

        let cols = v.len();
        let rows = v[0].len();

        let mut m: Matrix<T> = Matrix::new();
        
        for i in (0..cols) {
            if v[i].len() != rows {
                return None;
            }
            m = m.insert_column(m.cols(), &v[i]);
        }
        Some(m)
    }

    /// Creates a matrix from a vector of row vectors.
    pub fn from_row_vectors(v: &[Vec<T>]) -> Option<Matrix<T>> {

        if v.len() == 0 {
            return Some(Matrix::new());
        }

        let cols = v[0].len();

        let mut m: Matrix<T> = Matrix::new();

        for r in v.iter() {
            if r.len() != cols {
                return None;
            }
            m.add_row(r);
        }
        Some(m)
    }

    /// Creates a matrix from the values of an iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::Matrix;
    ///
    /// let v = vec![1, 2, 3, 4, 5, 6];
    /// let a = Matrix::from_it(v.iter(), 3).unwrap();
    /// assert_eq!(a.rows(), 2);
    /// assert_eq!(a.cols(), 3);
    /// ```
    pub fn from_it<I: Iterator<Item = T>>(iter: I, cols: usize) -> Option<Matrix<T>> {

        let v = iter.collect::<Vec<T>>();
        let l = v.len();
        Matrix::from_vec(v, l / cols, cols)
    }

    // TODO test
    pub fn from_file<R: FromStr + Clone>(fname: &str) -> Option<Matrix<R>> {

        match fs::File::open(fname) {
            Ok(mut f) => {
                let mut data = String::new();
                match f.read_to_string(&mut data) {
                    Ok(_n) => {
                        let mut m: Matrix<R> = Matrix::new();
                        for line in data.split("\n") {
                            let items = 
                                line.split(|c: char| c.is_whitespace()).filter(|s| s.len() > 0);
                            let mut v: Vec<R> = Vec::new();
                            for s in items {
                                match s.parse::<R>() {
                                    Ok(val) => v.push(val),
                                    _ => { return None; }
                                }
                            }
                            if v.len() > 0 {
                                if m.rows() > 0 && v.len() != m.cols() {
                                    return None;
                                }
                                m.add_row(&v);
                            }
                        }
                        Some(m)
                    }
                    _ => None
                }
            }
            _ => None
        }
    }

    /// Returns `true` if matrix has no rows and no columns, i.e.
    /// the matrix does not contain an element.
    ///
    /// # Example
    /// ```
    /// use rustml::Matrix;
    ///
    /// let m = Matrix::fill(1, 2, 2);
    /// assert!(!m.empty());
    /// let n = Matrix::<f64>::new();
    /// assert!(n.empty());
    /// ```
    pub fn empty(&self) -> bool {
        self.rows() == 0 && self.cols() == 0
    }

    /// Creates a matrix with random values.
    ///
    /// # Example
    /// ```
    /// use rustml::Matrix;
    ///
    /// let m = Matrix::<f64>::random::<f64>(3, 2);
    /// assert_eq!(m.rows(), 3);
    /// assert_eq!(m.cols(), 2);
    /// // all values are in the interval [0, 1)
    /// assert!(m.values().all(|&x| x < 1.0 && x >= 0.0));
    /// println!("{}", m);
    /// ```
    pub fn random<R: Rand + Clone>(rows: usize, cols: usize) -> Matrix<R> {

        let mut rng = thread_rng();
        Matrix::from_vec(
            rng.gen_iter::<R>().take(rows * cols).collect::<Vec<R>>(), 
            rows, cols
        ).unwrap()
    }

    // ------------------------------------

    /// Returns the internal buffer that is used to store the matrix.
    pub fn buf(&self) -> &Vec<T> { &self.data }

    /// Is equivalent to calling the method `cols()` on the matrix.
    pub fn lead_dim(&self) -> usize { self.cols()  }

    /// Returns the number of rows of the matrix.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5, 1.8; 
    ///     2.0, 2.5, 2.8
    /// ];
    /// assert_eq!(m.rows(), 2);
    /// # }
    /// ```
    pub fn rows    (&self) -> usize { self.nrows }
    
    /// Returns the number of columns of the matrix.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5, 1.8; 
    ///     2.0, 2.5, 2.8
    /// ];
    /// assert_eq!(m.cols(), 3);
    /// # }
    /// ```
    pub fn cols    (&self) -> usize { self.ncols }

    /// Returns an iterator over all elements of the matrix in row-major order.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5
    /// ];
    /// let mut i = m.values();
    /// assert_eq!(i.next().unwrap(), &1.0);
    /// assert_eq!(i.next().unwrap(), &1.5);
    /// assert_eq!(i.next().unwrap(), &2.0);
    /// # }
    /// ```
    pub fn values(&self) -> Iter<T> {
        self.data.iter()
    }

    /// Returns a mutable iterator over all elements of the matrix in row-major order.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let mut m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5
    /// ];
    /// for i in m.values_mut() {
    ///     *i = *i * 2.0;
    /// }
    /// assert!(m.eq(&mat![2.0, 3.0; 4.0, 5.0]));
    /// # }
    /// ```
    pub fn values_mut(&mut self) -> IterMut<T> {
        self.data.iter_mut()
    }

    /// Returns an iterator over the rows of the matrix.
    ///
    /// Each call to the `next` method is done in O(1).
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     5.0, 5.5
    /// ];
    /// let mut i = m.row_iter();
    /// assert_eq!(i.next().unwrap(), [1.0, 1.5]);
    /// assert_eq!(i.next().unwrap(), [2.0, 2.5]);
    /// assert_eq!(i.next().unwrap(), [5.0, 5.5]);
    /// # }
    /// ```
    pub fn row_iter(&self) -> RowIterator<T> {
        self.row_iter_at(0)
    }

    /// Returns an iterator over the rows of the matrix. The iterator
    /// starts at the the row with index `n`.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     5.0, 5.5
    /// ];
    /// let mut i = m.row_iter_at(1);
    /// assert_eq!(i.next().unwrap(), [2.0, 2.5]);
    /// assert_eq!(i.next().unwrap(), [5.0, 5.5]);
    /// # }
    /// ```
    pub fn row_iter_at(&self, n: usize) -> RowIterator<T> {

        RowIterator {
            m: self,
            idx: n
        }
    }

    /*
    pub fn row_iter_at_mut(&mut self, n: usize) -> RowIterMut<T> {

        RowIterMut {
            m: self,
            idx: n
        }
    }*/

    /// Returns an iterator over the rows of the matrix with the specified
    /// indexes in `rows`.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     3.0, 3.5;
    ///     4.0, 4.5;
    ///     5.0, 5.5
    /// ];
    /// let mut i = m.row_iter_of(&[1, 3, 4]);
    /// assert_eq!(i.next().unwrap(), [2.0, 2.5]);
    /// assert_eq!(i.next().unwrap(), [4.0, 4.5]);
    /// assert_eq!(i.next().unwrap(), [5.0, 5.5]);
    /// # }
    /// ```
    pub fn row_iter_of(&self, rows: &[usize]) -> SelectedRowIterator<T> {

        SelectedRowIterator {
            m: self,
            rows: rows.to_vec(),
            idx: 0,
        }
    }

    /// Returns the position where the element at row `row` and column `col`
    /// is stored in the internal vector that is used to store the matrix.
    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        
        match row < self.rows() && col < self.cols() {
            false => None,
            true  => Some(col + row * self.ncols)
        }
    }

    /// Returns the element of the matrix at row `row`
    /// (indexing starts at zero) and column `col` or
    /// `None` if row or column does not exist.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     5.0, 5.5
    /// ];
    /// assert_eq!(m.get(0, 1).unwrap(), &1.5);
    /// assert_eq!(m.get(2, 0).unwrap(), &5.0);
    /// assert!(m.get(3, 0).is_none());
    /// # }
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {

        match self.idx(row, col) {
            None => None,
            Some(p) => self.data.get(p)
        }
    }

    // TODO test
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {

        match self.idx(row, col) {
            None => None,
            Some(p) => self.data.get_mut(p)
        }
    }

    /// Returns the row at index `n` (starting at zero) in O(1).
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     5.0, 5.5
    /// ];
    /// assert_eq!(m.row(0).unwrap(), [1.0, 1.5]);
    /// assert_eq!(m.row(1).unwrap(), [2.0, 2.5]);
    /// assert_eq!(m.row(2).unwrap(), [5.0, 5.5]);
    /// assert!(m.row(3).is_none())
    /// # }
    /// ```
    pub fn row(&self, n: usize) -> Option<&[T]> {

        match self.idx(n, 0) {
            None => None,
            Some(p) => Some(self.data.split_at(p).1.split_at(self.ncols).0)
        }
    }

    /// Returns the row at index `n` in O(1) that is mutable.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let mut m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     5.0, 5.5
    /// ];
    /// {
    ///     let r = m.row_mut(1).unwrap();
    ///     r[0] = 4.0;
    ///     r[1] = 3.0;
    /// }
    /// assert_eq!(m.row(1).unwrap(), [4.0, 3.0]);
    /// # }
    /// ```
    pub fn row_mut(&mut self, n: usize) -> Option<&mut [T]> {

        match self.idx(n, 0) {
            None => None,
            Some(p) => Some(self.data.split_at_mut(p).1.split_at_mut(self.ncols).0)
        }
    }

    /// Replaces the element at row `row` (indexing starts at zero) and column `col` 
    /// with the new value `newval`. Returns true on
    /// success and false on failure, i.e. if the row or column does not exist.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let mut m = mat![
    ///     1.0, 1.5; 
    ///     2.0, 2.5;
    ///     5.0, 5.5
    /// ];
    /// assert_eq!(m.get(1, 0).unwrap(), &2.0);
    /// m.set(1, 0, 8.0);
    /// assert_eq!(m.get(1, 0).unwrap(), &8.0);
    /// assert_eq!(m.set(3, 0, 8.9), false);
    /// # }
    /// ```
    pub fn set(&mut self, row: usize, col: usize, newval: T) -> bool {

        match self.idx(row, col) {
            None => false,
            Some(p) => match self.data.get_mut(p) {
                Some(val) => {
                    *val = newval;
                    true
                }
                None => false,
            }
        }
    }

    pub fn map<F, U>(&self, f: F) -> Matrix<U>
        where F: FnMut(&T) -> U {

        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            data: self.data.iter().map(f).collect()
        }
    }

    pub fn add_row(&mut self, row: &[T]) {
        
        if self.rows() == 0 {
            self.ncols = row.len();
        } else {
            if self.cols() != row.len() {
                panic!("Invalid dimension.");
            }
        }
        self.nrows += 1;
        for i in row {
            self.data.push(i.clone());
        }
    }

    /// Extends the current matrix by putting the matrix `m` below it.
    pub fn push_matrix_below(&self, m: &Matrix<T>) -> Option<Matrix<T>> {

        if self.rows() > 0 && self.cols() != m.cols() {
            return None;
        }

        Some(Matrix {
            nrows: self.rows() + m.rows(),
            ncols: m.cols(),
            data: self.buf().iter().chain(m.buf().iter()).cloned().collect()
        })
    }

    // TODO tests
    pub fn find<F>(&self, f: F) -> Vec<(usize, usize)> 
        where F: Fn(&T) -> bool {

        self.values().enumerate()
            .filter(|&(_idx, val)| f(val))
            .map(|(idx, _val)| (idx % self.cols(), idx / self.cols()))
            .collect()
    }

    /// Inserts a column before the specified column (indexing starts at zero).
    ///
    /// If the matrix is empty a new matrix `n x 1` matrix is returned
    /// where `n` is the number of elements in `v`. If `pos`
    /// is greater or equal to the number columns the vector
    /// is appended as a new column.
    ///
    /// If the matrix is not empty and `v.len() != self.rows()` a
    /// `None` is returned.
    pub fn insert_column(&self, pos: usize, v: &[T]) -> Matrix<T> {

        if self.empty() {
            return Matrix::from_vec(v.to_vec(), v.len(), 1).unwrap();
        }

        assert!(v.len() == self.rows(), 
            "Length of vector must be equal to the number of rows."
        );

        let mut m = Matrix::<T>::new();
        for (i, r) in self.row_iter().enumerate() {
            let mut q = r.to_vec();
            if pos < q.len() {
                q.insert(pos, v[i].clone());
            } else {
                q.push(v[i].clone());
            }
            m.add_row(&q);
        }
        m
    }

    /// Returns a copy of the column at the specified index.
    pub fn column(&self, pos: usize) -> Option<Vec<T>> {

        if pos >= self.cols() {
            return None;
        }
        Some(self.row_iter().map(|r| r[pos].clone()).collect::<Vec<T>>())
    }

    /// Removes the column at index `pos` (indexing starts at zero)
    /// and returns the result.
    ///
    /// Panics if the column does not exist.
    pub fn rm_column(&self, pos: usize) -> Matrix<T> {

        assert!(pos < self.cols(), "Column does not exist.");

        // TODO a more efficient version

        let mut m = Matrix::new();
        if self.cols() > 1 {
            for r in self.row_iter() {
                let mut q = r.to_vec();
                q.remove(pos);
                m.add_row(&q);
            }
        }
        m
    }

    /// Iterates through the elements of the matrix, replaces elements
    /// and returns the new matrix.
    ///
    /// A element is replaced by `tr` if the predicate `prd` evaluated for this
    /// element returns `true`. If the predicate returns `false` the element
    /// is replaced by `fa`.
    ///
    /// The complexity is O(n) where `n` is the number of elements of the
    /// matrix.
    pub fn if_then_else<F>(&self, prd: F, tr: T, fa: T) -> Matrix<T> 
        where F: Fn(&T) -> bool {

        Matrix::from_it(
            self.values().map(|x| if prd(x) { tr.clone() } else { fa.clone() }), 
            self.cols()
        ).unwrap()
    }

    /// Iterates through the elements of the matrix and replaces the elements
    /// inplace.
    ///
    /// A element is replaced by `tr` if the predicate `prd` evaluated for this
    /// element returns `true`. If the predicate returns `false` the element
    /// is replaced by `fa`.
    ///
    /// The complexity is O(n) where `n` is the number of elements of the
    /// matrix.
    pub fn if_then_else_mut<F>(&mut self, prd: F, tr: T, fa: T)
        where F: Fn(&T) -> bool {

        for i in self.values_mut() {
            *i = if prd(i) { tr.clone() } else { fa.clone() };
        }
    }

    /// Reshapes the matrix, i.e. modifies the number of rows and columns.
    ///
    /// The complexity of this function is O(1) if `rows * cols` is equal
    /// to the number of elements in the matrix. If the new number of elements
    /// is smaller than the current number of elements the complexity is
    /// equal to the complexity of `Vec::truncate`.
    ///
    /// Panics if `rows * cols` is greater than the number of
    /// elements stored in the matrix.
    pub fn reshape_mut(&mut self, rows: usize, cols: usize) {

        assert!(rows * cols <= self.data.len(),
            "The new shape must not contain more elements."
        );

        if rows * cols < self.nrows * self.ncols {
            self.data.truncate(rows * cols);
        }
        self.nrows = rows;
        self.ncols = cols;
    }

    /// Reshapes the matrix, i.e. modifies the number of rows and columns
    /// and returns the result.
    ///
    /// The complexity of this function is O(rows * cols).
    ///
    /// Panics if `rows * cols` is greater than the number of
    /// elements stored in the matrix.
    pub fn reshape(&self, rows: usize, cols: usize) -> Matrix<T> {

        assert!(rows * cols <= self.data.len(),
            "The new shape must not contain more elements."
        );

        Matrix {
            nrows: rows,
            ncols: cols,
            data: self.data[..rows * cols].to_vec()
        }
    }
}

// --------------- Iterators ----------------------------------------

/// An iterator over the rows of a matrix.
pub struct RowIterator<'q, T: 'q> {
    m: &'q Matrix<T>,
    idx: usize
}

impl <'q, T: Clone> Iterator for RowIterator<'q, T> {
    type Item = &'q [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        match self.idx > self.m.rows() {
            true => None,
            false => self.m.row(self.idx - 1)
        }
    }
}

/*
/// A mutable iterator over the rows of a matrix.
pub struct RowIterMut<'q, T: 'q> {
    m: &'q mut Matrix<T>,
    idx: usize
}

impl <'q, T: Clone> Iterator for RowIterMut<'q, T> {
    type Item = &'q mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.idx += 1;
        match self.idx > self.m.rows() {
            true => None,
            false => self.m.row_mut(self.idx - 1)
        }
    }
}
*/

/// An iterator over a set of selected rows of a matrix.
pub struct SelectedRowIterator<'q, T: 'q> {
    m: &'q Matrix<T>,
    rows: Vec<usize>,
    idx: usize
}

impl <'q, T: Clone> Iterator for SelectedRowIterator<'q, T> {
    type Item = &'q [T];

    /// Returns the next row or `None` if the iterator has reached
    /// the end.
    fn next(&mut self) -> Option<Self::Item> {
        match self.idx < self.rows.len() {
            true => {
                self.idx += 1;
                self.m.row(*self.rows.get(self.idx - 1).unwrap())
            }
            false => None
        }
    }
}

// --------------- Matrix multiplication with BLAS ------------------

impl Mul for Matrix<f64> {
    type Output = Option<Matrix<f64>>;

    /// Performs a matrix multiplication by using the BLAS implementation
    /// that is linked with the binary.
    ///
    /// The complexity and performance of the operation depends on that
    /// implemenation. On failure the function returns None.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let a = mat![1.0f32, 2.0; 3.0, 4.0];
    /// let b = mat![4.0f32, 2.0; 5.0, 9.0];
    /// let c = (a * b).unwrap();
    /// assert_eq!(c.row(0).unwrap(), &[14.0, 20.0]);
    /// assert_eq!(c.row(1).unwrap(), &[32.0, 42.0]);
    /// println!("{}", c);
    /// # }
    /// ```
    fn mul(self, rhs: Matrix<f64>) -> Self::Output {

        if self.cols() != rhs.rows() {
            return None;
        }

        // TODO handling of NaN and stuff like this
        let c = Matrix::fill(0.0, self.rows(), rhs.cols());
        unsafe {
            cblas_dgemm(Order::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                self.rows()         as c_int,
                rhs.cols()          as c_int,
                self.cols()         as c_int,
                1.0                 as c_double,
                self.data.as_ptr()  as *const c_double,
                self.lead_dim()     as c_int,
                rhs.data.as_ptr()   as *const c_double,
                rhs.lead_dim()      as c_int,
                0.0                 as c_double,
                c.data.as_ptr()     as *mut c_double,
                c.lead_dim()        as c_int
            )
        }
        Some(c)
    }
}

// TODO test
impl Mul for Matrix<f32> {
    type Output = Option<Matrix<f32>>;

    /// Performs a matrix multiplication by using the BLAS implementation
    /// that is linked with the binary.
    ///
    /// The complexity and performance of the operation depends on that
    /// implemenation. On failure the function returns None.
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::*;
    ///
    /// # fn main() {
    /// let a = mat![1.0f32, 2.0; 3.0, 4.0];
    /// let b = mat![4.0f32, 2.0; 5.0, 9.0];
    /// let c = (a * b).unwrap();
    /// assert_eq!(c.row(0).unwrap(), [14.0, 20.0]);
    /// assert_eq!(c.row(1).unwrap(), [32.0, 42.0]);
    /// println!("{}", c);
    /// # }
    /// ```
    fn mul(self, rhs: Matrix<f32>) -> Self::Output {

        if self.cols() != rhs.rows() {
            return None;
        }

        // TODO handling of NaN and stuff like this
        let c = Matrix::<f32>::fill(0.0, self.rows(), rhs.cols());
        unsafe {
            cblas_sgemm(Order::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                self.rows()         as c_int,
                rhs.cols()          as c_int,
                self.cols()         as c_int,
                1.0                 as c_float,
                self.data.as_ptr()  as *const c_float,
                self.lead_dim()     as c_int,
                rhs.data.as_ptr()   as *const c_float,
                rhs.lead_dim()      as c_int,
                0.0                 as c_float,
                c.data.as_ptr()     as *mut c_float,
                c.lead_dim()        as c_int
            )
        }
        Some(c)
    }
}

// --------------- Matrix output ------------------------------------

impl <T: fmt::Display + Clone> fmt::Display for Matrix<T> {

    /// Implements `Display` so that the matrix can be printed with println!.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        for row in 0..self.rows() {
            for col in 0..self.cols() {
                // TODO use std::slice::SliceConcatExt if stable
                match write!(f, "{} ", self.get(row, col).unwrap()) {
                    Ok(_) => (),
                    e => return e,
                }
            }
            match writeln!(f, "") {
                Ok(_) => (),
                e => return e,
            }
        }
        write!(f, "")
    }
}

// --------------- Tests --------------------------------------------

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;
    use ops::MatrixScalarOps;

    #[test]
    fn test_from_vec() {

        let m = Matrix::from_vec(vec![1.0, 2.0], 2, 2);
        assert!(m.is_none());
        let p = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(p.is_some());
    }

    #[test]
    fn test_matrix() {

        let m: Matrix<f64> = Matrix::fill(1.0, 2, 3);

        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);

        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(*m.get(row, col).unwrap(), 1.0);
            }
        }

        assert!(m.get(2, 0).is_none());
        assert!(m.get(1, 0).is_some());
        assert!(m.get(1, 3).is_none());
    }

    #[test]
    fn test_mul1() {

        // [ 1 2 ]
        // [ 3 4 ]
        let va: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        // [ 5 ]
        // [ 6 ]
        let vb: Vec<f64> = vec![5.0, 6.0];

        let a: Matrix<f64> = Matrix::from_vec(va, 2, 2).unwrap();
        let b: Matrix<f64> = Matrix::from_vec(vb, 2, 1).unwrap();
        let c = (a * b).unwrap();

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 1);
        assert_eq!(*c.get(0, 0).unwrap(), 17.0);
        assert_eq!(*c.get(1, 0).unwrap(), 39.0);
    }

    #[test]
    fn test_mul2() {

        // [ 1 2 ]
        // [ 3 4 ]
        // [ 5 6 ]
        let va: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // [ 5 7 ]
        // [ 6 2 ]
        let vb: Vec<f64> = vec![5.0, 7.0, 6.0, 2.0];

        let a: Matrix<f64> = Matrix::from_vec(va, 3, 2).unwrap();
        let b: Matrix<f64> = Matrix::from_vec(vb, 2, 2).unwrap();
        let c = (a * b).unwrap();

        assert_eq!(c.rows(), 3);
        assert_eq!(c.cols(), 2);
        assert_eq!(*c.get(0, 0).unwrap(), 17.0);
        assert_eq!(*c.get(1, 0).unwrap(), 39.0);
        assert_eq!(*c.get(2, 0).unwrap(), 61.0);
        assert_eq!(*c.get(0, 1).unwrap(), 11.0);
        assert_eq!(*c.get(1, 1).unwrap(), 29.0);
        assert_eq!(*c.get(2, 1).unwrap(), 47.0);

        //         [ 17 11 ]
        // a * b = [ 39 29 ]
        //         [ 61 47 ]
    }

    #[test]
    fn test_macro() {

        let m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];
        assert_eq!(*m.get(0, 0).unwrap(), 1.0);
        assert_eq!(*m.get(0, 1).unwrap(), 2.0);
        assert_eq!(*m.get(1, 0).unwrap(), 3.0);
        assert_eq!(*m.get(1, 1).unwrap(), 4.0);
        assert_eq!(*m.get(2, 0).unwrap(), 5.0);
        assert_eq!(*m.get(2, 1).unwrap(), 6.0);

        let m2 = mat![1.0; 2.0];
        assert_eq!(m2.rows(), 2);
        assert_eq!(m2.cols(), 1);

        let m3 = mat![1.0, 2.0];
        assert_eq!(m3.rows(), 1);
        assert_eq!(m3.cols(), 2);
    }

    #[test]
    fn test_row() {

        let m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];
        assert_eq!(m.row(0).unwrap(), [1.0, 2.0]);
        assert_eq!(m.row(1).unwrap(), [3.0, 4.0]);
        assert_eq!(m.row(2).unwrap(), [5.0, 6.0]);
    }

    #[test]
    fn test_set() {

        let mut m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];
        assert_eq!(m.row(0).unwrap(), [1.0, 2.0]);
        assert_eq!(m.row(1).unwrap(), [3.0, 4.0]);
        assert_eq!(m.row(2).unwrap(), [5.0, 6.0]);
        m.set(0, 0, 7.0);
        m.set(2, 1, 9.0);
        assert_eq!(*m.get(0, 0).unwrap(), 7.0);
        assert_eq!(m.row(0).unwrap(), [7.0, 2.0]);
        assert_eq!(*m.get(2, 1).unwrap(), 9.0);
        assert_eq!(m.row(2).unwrap(), [5.0, 9.0]);
    }

    #[test]
    fn test_row_iter() {

        let m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];

        assert_eq!(m.row_iter().count(), 3);
        assert_eq!(m.row_iter().nth(0).unwrap(), [1.0, 2.0]);
        assert_eq!(m.row_iter().nth(1).unwrap(), [3.0, 4.0]);
        assert_eq!(m.row_iter().nth(2).unwrap(), [5.0, 6.0]);
    }

    #[test]
    fn test_row_iter_at() {

        let m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];

        assert_eq!(m.row_iter_at(1).count(), 2);
        assert_eq!(m.row_iter_at(1).nth(0).unwrap(), [3.0, 4.0]);
        assert_eq!(m.row_iter_at(1).nth(1).unwrap(), [5.0, 6.0]);
    }

    #[test]
    fn test_new() {

        let m: Matrix<f64> = Matrix::new();
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 0);
        assert!(m.get(0, 0).is_none());
        assert_eq!(m.row_iter().count(), 0);
    }

    #[test]
    fn test_random() {
        let m: Matrix<f64> = Matrix::<f64>::random::<f64>(3, 2);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.values().count(), 6);
        assert!(m.values().all(|x| *x >= 0.0 && *x <= 1.0));
    }

    #[test]
    fn test_row_iter_of() {
        let m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0; 7.0, 8.0];
        let v = [1, 3];
        let mut r = m.row_iter_of(&v);
        assert_eq!(r.next().unwrap(), [3.0, 4.0]);
        assert_eq!(r.next().unwrap(), [7.0, 8.0]);
        assert!(r.next().is_none());
    }

    #[test]
    fn test_has_nan() {
        let mut m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0; 7.0, 8.0];
        assert_eq!(m.has_nan(), false);
        m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0; 7.0, f64::NAN];
        assert!(m.has_nan());
    }

    #[test]
    fn test_map() {

        let y = mat![
            1u8, 2; 
            3, 4
        ].map(|&val| val as f32).mul_scalar(0.5);

        assert_eq!(y.buf(), &vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_row_mut() {

        let mut m = mat![1, 2; 3, 4];
        {
            let row = m.row_mut(0).unwrap();
            row[0] = 7;
            row[1] = 9;
        }
        {
            let row = m.row_mut(1).unwrap();
            row[0] = 6;
            row[1] = 11;
        }

        assert_eq!(m.row(0).unwrap().to_vec(), vec![7, 9]);
        assert_eq!(m.row(1).unwrap().to_vec(), vec![6, 11]);
    }

    #[test]
    fn test_add_row() {

        let mut m = Matrix::<i32>::new();
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 0);
        m.add_row(&vec![1, 2, 3]);
        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 3);
        m.add_row(&vec![4, 5, 6]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);

        assert_eq!(m.row(0).unwrap().to_vec(), vec![1, 2, 3]);
        assert_eq!(m.row(1).unwrap().to_vec(), vec![4, 5, 6]);
    }

    #[test]
    fn test_push_matrix_below() {

        let m = mat![
            1, 2, 3;
            4, 5, 6
        ];

        let a = mat![
            7, 8, 9;
            10, 11, 12
        ];

        let b = m.push_matrix_below(&a).unwrap();
        assert_eq!(b.rows(), 4);
        assert_eq!(b.row(0).unwrap().to_vec(), vec![1, 2, 3]);
        assert_eq!(b.row(1).unwrap().to_vec(), vec![4, 5, 6]);
        assert_eq!(b.row(2).unwrap().to_vec(), vec![7, 8, 9]);
        assert_eq!(b.row(3).unwrap().to_vec(), vec![10, 11, 12]);

        let c = mat![1, 2; 3, 4];
        assert!(c.push_matrix_below(&a).is_none());

        let d = Matrix::<i32>::new().push_matrix_below(&a).unwrap();
        assert_eq!(d.rows(), 2);
        assert_eq!(d.row(0).unwrap().to_vec(), vec![7, 8, 9]);
        assert_eq!(d.row(1).unwrap().to_vec(), vec![10, 11, 12]);
    }

    #[test]
    fn test_get_mut() {

        let mut a = mat![
            7, 8, 9;
            10, 11, 12
        ];

        {
            let v = a.get_mut(0, 2).unwrap();
            *v = 10;
        }
        assert_eq!(a.row(0).unwrap().to_vec(), vec![7, 8, 10]);
    }

    #[test]
    fn test_similar_matrix() {

        let a = mat![
            7.0, 8.0, 9.0;
            10.0, 11.0, 12.0
        ];

        let b = mat![
            9.0, 8.0, 9.0;
            10.0, 11.0, 12.0
        ];

        assert!(!a.similar(&b, 1.0)); 
        assert!(a.similar(&b, 2.0)); 
    }

    #[test]
    fn test_similar_vec() {

        let a = [1.0, 2.0, 3.0];
        assert!(a.similar(&[1.01, 1.99, 3.0], 0.0100001));
    }

    #[test]
    fn test_insert_column() {

        let a = mat![
            1, 2, 3;
            4, 5, 6
        ];
        assert_eq!(
            a.insert_column(0, &[8, 9]).buf(),
            &[8, 1, 2, 3, 9, 4, 5, 6]
        );
    }

    #[test]
    fn test_eq() {

        let a = mat![1.0, 2.0; 3.0, 4.0];
        let b = mat![2.0, 1.0; 3.0, 4.0];
        let c = mat![1.0, 2.0; 3.0, 4.0];
        assert!(a.eq(&c));
        assert!(!a.eq(&b));
    }

    #[test]
    fn test_from_col_vectors() {
        
        let v = [
            vec![1, 2, 3],
            vec![4, 5, 6]
        ];
        let m = Matrix::from_col_vectors(&v).unwrap();
        let e = mat![1,4; 2,5; 3,6];
        assert!(m.eq(&e));
    }

    #[test]
    fn test_from_row_vectors() {
        
        let v = [
            vec![1, 2, 3],
            vec![4, 5, 6]
        ];
        let m = Matrix::from_row_vectors(&v).unwrap();
        let e = mat![1,2,3; 4,5,6];
        assert!(m.eq(&e));
    }

    #[test]
    fn test_values_mut() {
        let mut m = mat![
            1.0, 1.5; 
            2.0, 2.5
        ];
        for i in m.values_mut() {
            *i = *i * 2.0;
        }
        assert!(m.eq(&mat![2.0, 3.0; 4.0, 5.0]));
    }

    #[test]
    fn test_from_it() {

        let v = vec![1, 2, 3, 4, 5, 6];
        let a = Matrix::from_it(v.iter(), 3).unwrap();
        assert_eq!(a.rows(), 2);
        assert_eq!(a.cols(), 3);
        let b = Matrix::from_it(v.iter(), 2).unwrap();
        assert_eq!(b.cols(), 2);
        assert_eq!(b.rows(), 3);
    }

    #[test]
    fn test_if_then_else() {

        let a = mat![1, 2, 3, 4; 3, 2, 4, 1];
        assert!(
            a.if_then_else(|&x| x > 2, 1, 0).eq(&mat![0, 0, 1, 1; 1, 0, 1, 0])
        );
    }

    #[test]
    fn test_if_then_else_mut() {

        let mut a = mat![
            1, 2, 3, 4; 
            3, 2, 4, 1
        ];
        a.if_then_else_mut(|&x| x > 2, 1, 0);
        assert!(a.eq(&mat![0, 0, 1, 1; 1, 0, 1, 0]));
    }

    #[test]
    fn test_to_matrix() {
        let v = vec![1, 2, 3, 4, 5];
        assert!(v.to_matrix(1).eq(&mat![1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_reshape_mut() {
        let mut a = mat![1, 2, 3, 4; 3, 2, 4, 1]; // 2x4
        a.reshape_mut(4, 2);
        assert_eq!(a.rows(), 4);
        assert_eq!(a.cols(), 2);
        assert!(a.eq(&mat![1, 2; 3, 4; 3, 2; 4, 1]));
    }

    #[test]
    fn test_reshape() {
        let a = mat![1, 2, 3, 4; 3, 2, 4, 1]; // 2x4
        let b = a.reshape(4, 2);
        assert_eq!(b.rows(), 4);
        assert_eq!(b.cols(), 2);
        assert!(b.eq(&mat![1, 2; 3, 4; 3, 2; 4, 1]));
    }

    #[test]
    fn test_from_iter() {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let m = v.iter().cloned().collect::<Matrix<_>>();
        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 8);
        assert!(m.eq(&mat![1, 2, 3, 4, 5, 6, 7, 8]));
    }
}

