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
//! This module provides low level functions to access the BLAS functions. It is highly recommended
//! to use the wrappers in the module [ops_inplace](../ops_inplace/index.html) which provide
//! a more convenient and safer high level interface.
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
    /// The paramters `alpha` is a scalar of type f64 and `x` and `y` are 
    /// vectors with elements of type f64. The parameter `n` specifies
    /// the number of elements in `x` and `y`. The parameters `incx`
    /// and `incy` specify the increments between the elements in 
    /// vector `x` and `y` respectively.
    ///
    /// For a high level interface you should use [d_axpy](../ops_inplace/fn.d_axpy.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
    pub fn cblas_daxpy(
        n: c_int, 
        alpha: c_double, 
        x: *const c_double, 
        incx: c_int, 
        y: *mut c_double, 
        incy: c_int
    );

    /// Computes `alpha * op(A) * op(B) + beta * C` and stores the result in `C`.
    ///
    /// The parameters `alpha` and `beta` are scalars of type `f64`, `A`, `B`, `C` are a
    /// matrices with elements of type `f64` and `op(X)` is either
    /// `op(X) = X` or `op(X) = X^T` (the transpose or conjugate transpose of
    /// the matrix `X`).
    ///
    /// For a high level interface you should use [d_gemm](../ops_inplace/fn.d_gemm.html)
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

    /// Computes `alpha * A * x + beta * y` or `alpha * A^T * x + beta * y` and stores the
    /// result in `y`.
    ///
    /// The parameter `order` specifies the memory layout of the matrix `A`. Matrices 
    /// in rustml are stored in [`RowMajor`](enum.Order.html) order by default. If the parameter `transA`
    /// is set to [`Trans`](enum.Transpose.html) the transpose of `A` is used, otherwise `A`. The parameter
    /// `m` specifies the number of rows of `A`, `n` the number of columns, `lda` should be
    /// set to the number of columns of `A`.
    /// 
    /// For a high level interface you should use [d_gemv](../ops_inplace/fn.d_gemv.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
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

    /// Computes the L2 norm (euclidean norm) of a vector of elements of type f64 (doubles).
    ///
    /// The parameter `n` specifies the number of elements in the vector `x`. The parameter
    /// `incx` specifies the increment between the elements of `x`.
    ///
    /// For a high level interface you should use [d_nrm2](../ops_inplace/fn.d_nrm2.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
    pub fn cblas_dnrm2(n: c_int, x: *const c_double, incx: c_int) -> c_double;

    /// Computes `alpha * x + y` and stores the result in `y`.
    ///
    /// The paramters `alpha` is a scalar of type f32 and `x` and `y` are 
    /// vectors with elements of type f32. The parameter `n` specifies
    /// the number of elements in `x` and `y`. The parameters `incx`
    /// and `incy` specify the increments between the elements in 
    /// vector `x` and `y` respectively.
    ///
    /// For a high level interface you should use [s_axpy](../ops_inplace/fn.s_axpy.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
    pub fn cblas_saxpy(
        n: c_int, 
        alpha: c_float, 
        x: *const c_float, 
        incx: c_int, 
        y: *mut c_float, 
        incy: c_int
    );

    /// Computes `alpha * op(A) * op(B) + beta * C` and stores the result in `C`.
    ///
    /// The parameters `alpha` and `beta` are scalars of type `f32`, `A`, `B`, `C` are a
    /// matrices with elements of type `f32` and `op(X)` is either
    /// `op(X) = X` or `op(X) = X^T` (the transpose or conjugate transpose of
    /// the matrix `X`).
    ///
    /// For a high level interface you should use [s_gemm](../ops_inplace/fn.s_gemm.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
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


    /// Computes the L2 norm (euclidean norm) of a vector of elements of type f32 (floats
    ///
    /// The parameter `n` specifies the number of elements in the vector `x`. The parameter
    /// `incx` specifies the increment between the elements of `x`.
    ///
    /// For a high level interface you should use [s_nrm2](../ops_inplace/fn.s_nrm2.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
    pub fn cblas_snrm2(n: c_int, x: *const c_float, incx: c_int) -> c_float;

    /// Computes `alpha * A * x + beta * y` or `alpha * A^T * x + beta * y` and stores the
    /// result in `y`.
    ///
    /// The parameter `order` specifies the memory layout of the matrix `A`. Matrices 
    /// in rustml are stored in [`RowMajor`](enum.Order.html) order by default. If the parameter `transA`
    /// is set to [`Trans`](enum.Transpose.html) the transpose of `A` is used, otherwise `A`. The parameter
    /// `m` specifies the number of rows of `A`, `n` the number of columns, `lda` should be
    /// set to the number of columns of `A`.
    /// 
    /// For a high level interface you should use [s_gemv](../ops_inplace/fn.s_gemv.html)
    /// in the module [ops_inplace](../ops_inplace/index.html).
    pub fn cblas_sgemv(
        order: Order, transA: Transpose,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float, lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int
    );
}

