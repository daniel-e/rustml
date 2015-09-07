//! Hash functions.

use std::u32;

/// A simple hash functions.
///
/// <script type="text/javascript"
///   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
/// </script>
/// <script type="text/x-mathjax-config">
///   MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
/// </script>
///
/// This functions computes a hash as follows: $h(s) = \sum_{i=0}\^n 31\^{n-i} s_i $
pub fn simple_hash(s: &[u8]) -> u32 {

    let m = u32::MAX as u64;
    s.iter().fold::<u64, _>(0, |acc, x| (acc * 31 + (*x as u64)) & m) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        assert_eq!(simple_hash("a".as_bytes()), 97);
        assert_eq!(simple_hash("Joe Miller".as_bytes()), 149190249);
    }
}

