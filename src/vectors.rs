extern crate libc;

use std::cmp::{PartialEq, min};
use self::libc::{c_void, size_t};
use std::mem;

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


extern {
    fn memcpy(dst: *mut c_void, src: *const c_void, n: size_t);
}

/// Copies elements from `src` to `dst`.
pub fn copy_memory<T>(dst: &mut [T], src: &[T], n: usize) -> usize {

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
    use super::{group, copy_memory};

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

