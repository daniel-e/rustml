extern crate libc;

use self::libc::{c_int, c_double};

// documentation
// http://www.netlib.org/blas/
// file::///usr/include/cblas.h

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
    /// Computes the L2 norm of a vector of f64 (doubles).
    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    /// Computes alpha * x + y where alpha is a f64 scalar and x and y are 
    /// vectors of f64.
    pub fn cblas_daxpy(
        n: c_int, 
        alpha: c_double, 
        x: *const c_double, 
        incx: c_int, 
        y: *mut c_double, 
        incy: c_int
    );

    /// Computes alpha * op(A) * op(B) + beta * C where op(X) is either
    /// op(X) = X or op(X) = X^T (the transpose of the matrix X).
    pub fn cblas_dgemm(
        order: Order, transA: Transpose, transB: Transpose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        A: *const c_double, lda: c_int,
        B: *const c_double, ldb: c_int,
        beta: c_double,
        C: *mut c_double, ldc: c_int
    );
}

