#![macro_use]

extern crate libc;
extern crate rand;

use std::{iter, fmt};
use std::ops::Mul;
use ::blas::{Order, Transpose, cblas_dgemm};
use self::rand::{thread_rng, Rng, Rand};
use std::slice::Iter;

pub struct Matrix<T> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>
}

impl <T: Clone> Matrix<T> {

    // Functions for constructing a matrix.

    pub fn new() -> Matrix<T> {

        Matrix::from_vec(Vec::new(), 0, 0).unwrap()
    }

    pub fn fill(value: T, rows: usize, cols: usize) -> Matrix<T> {

        Matrix::from_vec(
            iter::repeat(value).take(rows * cols).collect(),
            rows, cols
        ).unwrap()
    }

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

    /// Creates a matrix with random values.
    ///
    /// # Example
    /// ```
    /// let m = Matrix::<f64>::random::<f64>(3, 2);
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

    pub fn lead_dim(&self) -> usize { self.cols()  }
    pub fn rows    (&self) -> usize { self.nrows   }
    pub fn cols    (&self) -> usize { self.ncols   }

    pub fn values(&self) -> Iter<T> {
        self.data.iter()
    }

    /// Each call of the iterator's next() method is O(1).
    pub fn row_iter(&self) -> RowIterator<T> {
        self.row_iter_at(0)
    }

    pub fn row_iter_at(&self, row: usize) -> RowIterator<T> {

        RowIterator {
            m: self,
            idx: row
        }
    }

    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        
        match row < self.rows() && col < self.cols() {
            false => None,
            true  => Some(col + row * self.ncols)
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {

        match self.idx(row, col) {
            None => None,
            Some(p) => self.data.get(p)
        }
    }

    pub fn row(&self, row: usize) -> Option<&[T]> {

        match self.idx(row, 0) {
            None => None,
            Some(p) => Some(self.data.split_at(p).1.split_at(self.ncols).0)
        }
    }

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

}

// --------------- Iterator -----------------------------------------

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

// --------------- Matrix multiplication with BLAS ------------------

impl Mul for Matrix<f64> {
    type Output = Option<Matrix<f64>>;

    fn mul(self, rhs: Matrix<f64>) -> Self::Output {

        if self.cols() != rhs.rows() {
            return None;
        }

        // TODO handling of NaN and stuff like this
        let c = Matrix::fill(0.0, self.rows(), rhs.cols());
        unsafe {
            cblas_dgemm(Order::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                self.rows()         as libc::c_int,
                rhs.cols()          as libc::c_int,
                self.cols()         as libc::c_int,
                1.0                 as libc::c_double,
                self.data.as_ptr()  as *const libc::c_double,
                self.lead_dim()     as libc::c_int,
                rhs.data.as_ptr()   as *const libc::c_double,
                rhs.lead_dim()      as libc::c_int,
                0.0                 as libc::c_double,
                c.data.as_ptr()     as *mut libc::c_double,
                c.lead_dim()        as libc::c_int
            )
        }
        Some(c)
    }
}

// --------------- Matrix output ------------------------------------

impl <T: fmt::Display + Clone> fmt::Display for Matrix<T> {

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

// --------------- Matrix macro mat! --------------------------------

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

// --------------- Tests --------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}

