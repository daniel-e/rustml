#![macro_use]

extern crate libc;

use std::{iter, fmt};
use std::ops::Mul;
use ::blas::{Order, Transpose, cblas_dgemm};


pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>
}


pub trait MatrixOps<T> {

    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn lead_dim(&self) -> usize;
    fn data(&self) -> &Vec<T>;
    fn get(&self, row: usize, col: usize) -> Option<&T>;
    fn row(&self, idx: usize) -> Option<Vec<T>>;
    fn set(&mut self, row: usize, col: usize, newval: T);
}

impl <T: Clone> MatrixOps<T> for Matrix<T> {

    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn lead_dim(&self) -> usize { self.cols() }
    fn data(&self) -> &Vec<T> { &self.data }

    fn get(&self, row: usize, col: usize) -> Option<&T> {

        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get(col + row * self.cols)
    }

    fn row(&self, idx: usize) -> Option<Vec<T>> {

        if idx >= self.rows {
            return None;
        }
        let mut v = Vec::new();
        for i in 0..self.cols {
            v.push(self.get(idx, i).unwrap().clone());
        }
        Some(v)
    }

    fn set(&mut self, row:usize, col: usize, newval: T) {

        if row < self.rows() && col < self.cols() {

            match self.data.get_mut(col + row * self.cols) {
                Some(ref mut val) => {
                    **val = newval; // TODO ??
                }
                _ => (),
            }
        }
    }
}

impl <T: Clone> Matrix<T> {

    pub fn fill(value: T, rows: usize, cols: usize) -> Matrix<T> {

        Matrix {
            rows: rows,
            cols: cols,
            data: iter::repeat(value).take(rows * cols).collect()
        }
    }

    pub fn from_vec(vals: &Vec<T>, rows: usize, cols: usize) -> Option<Matrix<T>> {

        if rows * cols != vals.len() {
            return None;
        }

        Some(Matrix {
            rows: rows,
            cols: cols,
            data: vals.clone()
        })
    }

}

impl Mul for Matrix<f64> {
    type Output = Matrix<f64>;

    fn mul(self, rhs: Matrix<f64>) -> Matrix<f64> {

        if self.cols() != rhs.rows() {
            panic!("Columns of left hand side must be equal to rows of right hand side.");
        }

        let c: Matrix<f64> = Matrix::fill(0.0, self.rows(), rhs.cols());

        unsafe {
            cblas_dgemm(Order::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
                self.rows()         as libc::c_int,
                rhs.cols()          as libc::c_int,
                self.cols()         as libc::c_int,
                1.0                 as libc::c_double,
                self.data.as_ptr()  as *const libc::c_double,
                self.lead_dim()     as libc::c_int,
                rhs.data().as_ptr() as *const libc::c_double,
                rhs.lead_dim()      as libc::c_int,
                0.0                 as libc::c_double,
                c.data().as_ptr()   as *mut libc::c_double,
                c.lead_dim()        as libc::c_int
            )
        }
        c
    }
}

impl <T: fmt::Display> fmt::Display for Matrix<T> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        for row in 0..self.rows {
            for col in 0..self.cols {
                match write!(f, "{} ", self.data.get(col + row * self.cols).unwrap()) {
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
        Matrix::from_vec(&v, rows, cols_old).unwrap()
        }
    };
}


#[cfg(test)]
mod tests {
    use super::*;

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

        let a: Matrix<f64> = Matrix::from_vec(&va, 2, 2).unwrap();
        let b: Matrix<f64> = Matrix::from_vec(&vb, 2, 1).unwrap();
        let c = a * b;

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

        let a: Matrix<f64> = Matrix::from_vec(&va, 3, 2).unwrap();
        let b: Matrix<f64> = Matrix::from_vec(&vb, 2, 2).unwrap();
        let c = a * b;

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
        let v1 = vec![1.0, 2.0];
        let v2 = vec![3.0, 4.0];
        let v3 = vec![5.0, 6.0];
        assert_eq!(m.row(0).unwrap(), v1);
        assert_eq!(m.row(1).unwrap(), v2);
        assert_eq!(m.row(2).unwrap(), v3);
    }

    #[test]
    fn test_set() {

        let mut m = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];
        let v1 = vec![1.0, 2.0];
        let v2 = vec![3.0, 4.0];
        let v3 = vec![5.0, 6.0];
        assert_eq!(m.row(0).unwrap(), v1);
        assert_eq!(m.row(1).unwrap(), v2);
        assert_eq!(m.row(2).unwrap(), v3);
        m.set(0, 0, 7.0);
        m.set(2, 1, 9.0);
        assert_eq!(*m.get(0, 0).unwrap(), 7.0);
        assert_eq!(m.row(0).unwrap(), vec![7.0, 2.0]);
        assert_eq!(*m.get(2, 1).unwrap(), 9.0);
        assert_eq!(m.row(2).unwrap(), vec![5.0, 9.0]);
    }
}

