#![macro_use]

extern crate libc;

use std::iter;
use std::fmt;
use std::ops::Mul;


#[repr(C)]
pub enum Order {
    RowMajor = 101,
    ColMajor = 102
}

#[repr(C)]
pub enum Transpose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113
}

#[link(name = "blas")]
extern {
    fn cblas_dgemm(order: Order, transA: Transpose, transB: Transpose,
            m: libc::c_int,
            n: libc::c_int,
            k: libc::c_int,
            alpha: libc::c_double,
            A: *const libc::c_double,
            lda: libc::c_int,
            B: *const libc::c_double,
            ldb: libc::c_int,
            beta: libc::c_double,
            C: *mut libc::c_double,
            ldc: libc::c_int
    );
}

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
    fn add_row(&mut self, row: Vec<T>);
}

impl <T: Clone> MatrixOps<T> for Matrix<T> {

    fn rows(&self) -> usize { self.rows }
    fn cols(&self) -> usize { self.cols }
    fn lead_dim(&self) -> usize { self.rows() }
    fn data(&self) -> &Vec<T> { &self.data }

    fn get(&self, row: usize, col: usize) -> Option<&T> {

        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data.get(row + col * self.rows)
    }

    fn add_row(&mut self, row: Vec<T>) {

        if self.rows() == 0 && self.cols() == 0 {
            self.cols = row.len();
            for i in row {
                self.data.push(i);
            }
            self.rows = 1;
        } else {
            let mut pos = self.rows();
            for i in row {
                self.data.insert(pos, i);
                pos += self.rows() + 1;
            }
            self.rows += 1;
        }
    }
}

impl <T: Clone> Matrix<T> {

    pub fn new() -> Matrix<T> {

        Matrix {
            rows: 0,
            cols: 0,
            data: Vec::new()
        }
    }

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
            cblas_dgemm(Order::ColMajor, Transpose::NoTrans, Transpose::NoTrans,
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
                match write!(f, "{} ", self.data.get(row + col * self.rows).unwrap()) {
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
        let mut m = Matrix::new();
        $(
            let mut v = Vec::new();
            $(
                v.push($x);
            )+
            m.add_row(v);
        )*
        m
        }
    };
}


#[cfg(test)]
mod tests {
    use super::{Matrix, MatrixOps};

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
        let va: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0];

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
        let va: Vec<f64> = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];

        // [ 5 7 ]
        // [ 6 2 ]
        let vb: Vec<f64> = vec![5.0, 6.0, 7.0, 2.0];

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

        println!("{}", c);
        //         [ 17 11 ]
        // a * b = [ 39 29 ]
        //         [ 61 47 ]
    }

    #[test]
    fn test_macro() {

        let m = mat![1, 2; 3, 4; 5, 6];
        println!("{}", m);
    }
}

