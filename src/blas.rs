extern crate libc;

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
    // computes L2 norm of doubles
    pub fn cblas_dnrm2(n: libc::c_int, x: *const libc::c_double, incx: libc::c_int) -> libc::c_double;

    // computes alpha * x + y of vectors x and y of doubles
    pub fn cblas_daxpy(n: libc::c_int, alpha: libc::c_double, 
                   x: *const libc::c_double, incx: libc::c_int, 
                   y: *mut libc::c_double, incy: libc::c_int);

    // computes alpha * op(A) * op(B) + beta * C
    // where op(X) is either op(X) = X or op(X) = X^T
    pub fn cblas_dgemm(order: Order, transA: Transpose, transB: Transpose,
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


