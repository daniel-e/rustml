//! Bindings for BLAS/ATLAS for high performance vector and matrix operations.

extern crate libc;

use self::libc::{c_int, c_double, c_float};

// documentation
// http://www.netlib.org/blas/
// file::///usr/include/cblas.h

/// Enum to specify how a matrix is arranged. Required for the
/// `cblas_*` functions.
#[repr(C)]
pub enum Order {
    /// row-major order
    RowMajor = 101,
    /// column-major order
    ColMajor = 102
}

/// Enum to specify how to transform a matrix before doing an
/// operation on it. Required for the `cblas_*` functions.
#[repr(C)]
pub enum Transpose {
    /// No transformation of the matrix.
    NoTrans = 111,
    /// Use the transpose of the matrix.
    Trans = 112,
    /// Use the conjugate transpose of the matrix.
    ConjTrans = 113
}

#[link(name = "blas")]
extern {
    // TODO wrapper functions

    /// Computes the L2 norm of a vector of f64 (doubles).
    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    /// Computes the L2 norm of a vector of f32 (floats).
    pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;

    /// Computes `alpha * x + y` where `alpha` is a f64 scalar and `x` and `y` are 
    /// vectors of f64.
    pub fn cblas_daxpy(
        n: c_int, 
        alpha: c_double, 
        x: *const c_double, 
        incx: c_int, 
        y: *mut c_double, 
        incy: c_int
    );

    /// Computes `alpha * x + y` where `alpha` is a f64 scalar and `x` and `y` are 
    /// vectors of f32.
    pub fn cblas_saxpy(
        n: c_int, 
        alpha: c_float, 
        x: *const c_float, 
        incx: c_int, 
        y: *mut c_float, 
        incy: c_int
    );

    /// Computes `alpha * op(A) * op(B) + beta * C` where `alpha` and
    /// `beta` are f64 scalars, `A`, `B`, `C` are a matrices of f64 values and `op(X)` is either
    /// `op(X) = X` or `op(X) = X^T` (the transpose or conjugate transpose of
    /// the matrix `X`).
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

    /// Computes `alpha * op(A) * op(B) + beta * C` where `alpha` and
    /// `beta` are f32 scalars, `A`, `B`, `C` are a matrices of f32 values and `op(X)` is either
    /// `op(X) = X` or `op(X) = X^T` (the transpose or conjugate transpose of
    /// the matrix `X`).
    pub fn cblas_sgemm(
        order: Order, transA: Transpose, transB: Transpose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        A: *const c_float, lda: c_int,
        B: *const c_float, ldb: c_int,
        beta: c_float,
        C: *mut c_float, ldc: c_int
    );
}

