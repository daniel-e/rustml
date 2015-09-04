//! Bindings for BLAS/ATLAS for high performance vector and matrix operations.
//!
//! from [here](http://www.netlib.org/blas/):
//!
//! <i>The BLAS (Basic Linear Algebra Subprograms) are routines that provide standard building blocks
//! for performing basic vector and matrix operations. The Level 1 BLAS perform scalar, vector and
//! vector-vector operations, the Level 2 BLAS perform matrix-vector operations, and the Level 3
//! BLAS perform matrix-matrix operations. Because the BLAS are efficient, portable, and widely
//! available, they are commonly used in the development of high quality linear algebra
//! software.</i>
//!
//! There are several implementations of BLAS:
//!
//! * [OpenBLAS](http://www.openblas.net/)
//! * [ATLAS (automatically tuned linear algebra software)](http://math-atlas.sourceforge.net/)
//! * [Netlib BLAS](http://www.netlib.org/blas/)
//! * [Intel MKL](https://software.intel.com/en-us/intel-mkl)
//! * and some more
//!
//! The Netlib BLAS implementation (reference implementation) is usually required when
//! compiling rustml but you can switch to
//! any of these implementations without recompiling rustml simply by setting the
//! `LD_PRELOAD` environment variable to the location of the library you want to use
//! before running your application.
//!
//! Example: Let's assume you have installed ATLAS into `/opt/atlas` and want to start you
//! application with cargo. Then, you can use the
//! ATLAS implementation simply by starting your application as follows (depending on your
//! installation):
//!
//! ```ignore
//! LD_PRELOAD=/opt/atlas/lib/libtatlas.so cargo run myapp
//! ```
//!
//! # Using BLAS for vector and matrix operations
//!
//! This module provides low level functions. ...
//! 
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

    /// Computes `alpha * x + y` and stores the result in `y`.
    ///
    /// The paramters `alpha` is a f64 scalar and `x` and `y` are 
    /// vectors with elements of type f64. The parameter `n` specifies
    /// the number of elements in `x` and `y`. The parameters `incx`
    /// and `incy` specify the increments between the elements in 
    /// vector `x` and `y` respectively.
    ///
    /// For a high level abstraction you should use [d_axpy](../ops_inplace/fn.d_axpy.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
    pub fn cblas_daxpy(
        n: c_int, 
        alpha: c_double, 
        x: *const c_double, 
        incx: c_int, 
        y: *mut c_double, 
        incy: c_int
    );

    /// Computes `alpha * op(A) * op(B) + beta * C` and stores the result in `C` where `alpha` and
    /// `beta` are f64 scalars, `A`, `B`, `C` are a matrices of f64 values and `op(X)` is either
    /// `op(X) = X` or `op(X) = X^T` (the transpose or conjugate transpose of
    /// the matrix `X`).
    ///
    /// For a high level abstraction you should use [d_gemm](../ops_inplace/fn.d_gemm.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
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


    /// Computes the L2 norm of a vector of f64 (doubles).
    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    /// Computes the L2 norm of a vector of f32 (floats).
    pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;

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

    pub fn cblas_dgemv(
        order: Order, transA: Transpose,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double, lda: c_int,
        x: *const c_double,
        incx: c_int,
        beta: c_double,
        y: *mut c_double,
        incy: c_int
    );
}

