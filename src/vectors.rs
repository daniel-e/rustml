//! Functions to operate on vectors.
extern crate libc;
extern crate num;
extern crate rand;

use std::cmp::{PartialEq, min};
use self::libc::{c_void, size_t};
use std::marker::Copy;
use self::num::traits::Num;
use std::iter;
use self::rand::{thread_rng, Rng, Rand};
use std::mem;

// ------------------------------------------------------------------

use std::io::{Read, BufRead, BufReader, Result, Error, ErrorKind};
use std::fs::File;
use std::str::FromStr;
use std::marker::PhantomData;

pub struct VecReader<B, T> {
    buf: B,
    phantom: PhantomData<T>
}

impl <T: FromStr, B: BufRead> Iterator for VecReader<B, T> {
    type Item = Result<Vec<T>>;

    fn next(&mut self) -> Option<Result<Vec<T>>> {

        let mut s = String::new();
        match self.buf.read_line(&mut s) {
            Ok(0)  => None,
            Ok(_n) => {
                if s.ends_with("\n") {
                    s.pop();
                }
                let mut ve: Vec<T> = Vec::new();
                for i in s.split(" ") {
                    match i.parse::<T>() {
                        Ok(i) => ve.push(i),
                        Err(_e) => return 
                            Some(
                                Err(Error::new(ErrorKind::InvalidInput, "parse error"))
                            )
                    }
                }
                Some(Ok(ve))
            }
            Err(e) => Some(Err(e))
        }
    }
}

pub fn from_reader<T: FromStr, R: Read>(f: BufReader<R>) -> VecReader<BufReader<R>, T> {

    VecReader {
        buf: f,
        phantom: PhantomData
    }
}

pub fn from_file<T: FromStr>(path: &str) -> Result<VecReader<BufReader<File>, T>> {

    Ok(VecReader {
        buf: BufReader::<File>::new(try!(File::open(path))),
        phantom: PhantomData
    })
}

// ------------------------------------------------------------------

/// Groups equal elements into one element and counts them.
pub fn group<T: PartialEq + Clone>(v: &Vec<T>) -> Vec<(T, usize)> {

    let mut r: Vec<(T, usize)> = Vec::new();
    for val in v {
        if r.len() == 0 {
            r.push((val.clone(), 1));
        } else {
            let mut x = r.pop().unwrap();
            if x.0 != *val {
                r.push(x);
                x = (val.clone(), 0);
            }
            x.1 += 1;
            r.push(x);
        }
    }
    r
}

// ------------------------------------------------------------------

/// Creates a vector for which all elements are equal to zero.
///
/// # Example
///
/// ```
/// use rustml::vectors::zero;
///
/// let b = zero::<i32>(5);
/// assert_eq!(b, vec![0, 0, 0, 0, 0]);
/// let c = zero::<i32>(0);
/// assert_eq!(c, vec![]);
/// ```
pub fn zero<T: Num + Clone>(n: usize) -> Vec<T> {

    // TODO more efficient implementation
    iter::repeat(T::zero()).take(n).collect()
}

// ------------------------------------------------------------------

/// Creates a vector with random elements.
///
/// # Example
///
// TODO
pub fn random<T: Rand + Clone>(n: usize) -> Vec<T> {

    thread_rng().gen_iter::<T>().take(n).collect::<Vec<T>>()
}

// ------------------------------------------------------------------

extern {
    fn memcpy(dst: *mut c_void, src: *const c_void, n: size_t);
}

/// Copies elements from `src` to `dst`.
///
/// # Implementation details
///
/// This function uses the C function call `memcpy` to copy the memory.
pub fn copy_memory<T: Copy>(dst: &mut [T], src: &[T], n: usize) -> usize {

    let c = min(min(dst.len(), src.len()), n);
    unsafe {
        memcpy(
            dst.as_ptr()              as *mut c_void, 
            src.as_ptr()              as *const c_void,
            (c * mem::size_of::<T>()) as size_t
        );
    }
    c
}

#[cfg(test)]
mod tests {
    extern crate num;
    use super::*;

    #[test]
    fn test_zero() {

        let a = zero::<f32>(4);
        assert_eq!(a, vec![0.0, 0.0, 0.0, 0.0]);
        let b = zero::<i32>(5);
        assert_eq!(b, vec![0, 0, 0, 0, 0]);
        let c = zero::<i32>(0);
        assert_eq!(c, vec![]);
    }

    #[test]
    fn test_group() {

        let mut v = vec![1.0, 1.0, 2.0, 7.0, 7.0, 9.0, 9.0, 9.0];
        let mut r = group(&v);
        assert_eq!(r, vec![(1.0, 2), (2.0, 1), (7.0, 2), (9.0, 3)]);

        v = vec![];
        r = group(&v);
        assert_eq!(r, vec![]);

        v = vec![1.0, 2.0, 2.0, 2.0, 3.0, 4.0];
        r = group(&v);
        assert_eq!(r, vec![(1.0, 1), (2.0, 3), (3.0, 1), (4.0, 1)]);
    }

    #[test]
    fn test_copy_memory() {

        let mut a = [0, 0, 0, 0];
        let b = [1, 2, 3, 4];
        assert_eq!(copy_memory(&mut a, &b, 4), 4);
        assert_eq!(a, b);

        assert_eq!(copy_memory(&mut a, &b, 5), 4);

        let mut c = [1.0, 2.0, 3.0];
        let d = [5.0, 6.0, 7.9];
        assert_eq!(copy_memory(&mut c, &d, 3), 3);
        assert_eq!(c, d);
    }
}

