
use std::cmp::Ordering;
use ::matrix::Matrix;

/// Searches for the k nearest neighbours for a given row in all rows
/// (except the one indexed by idx) of the given matrix.
///
/// If m is the number of rows and n is the number of columns the
/// complexity is O(mn + m log m).
pub fn knn_scan_1<D>(m: &Matrix<f64>, idx: usize, k: usize, df: D) -> Option<Vec<(usize, f64)>> 
    where D : Fn(&[f64], &[f64]) -> f64 {

    let row = m.row(idx).expect("Could not fetch row.");

    let mut v = Vec::new();

    for i in 0..m.rows() { // O(m)
        if i != idx {
            let d = df(&m.row(i).unwrap(), &row); // O(n)
            v.push((i, d));
        }
    }

    // TODO http://doc.rust-lang.org/std/cmp/trait.PartialOrd.html
    // NaN

    //O(m log m)
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    Some(v.iter().take(k).cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::knn_scan_1;
    use ::matrix::*;
    use :: distance::*;

    #[test]
    fn test_knn_scan_1() {

        let m = mat![1.0, 2.0; 3.0, 3.0; 1.0, 2.0; 5.0, 6.0; 10.0, 12.0];

        // 1st nearest neighbours of [3.0, 3.0]
        let mut r = knn_scan_1(&m, 1, 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(r.len(), 1);
        r.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(r.get(0).unwrap().0, 0);

        // 2-nearest neighbours of [3.0, 3.0] (= [1.0, 2.0], [1.0, 2.0])
        r = knn_scan_1(&m, 1, 2, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(r.len(), 2);
        r.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(r.get(0).unwrap().0, 0);
        assert_eq!(r.get(1).unwrap().0, 2);

        // 1-nearest neighbour of [1.0, 2.0] (= [1.0, 2.0])
        r = knn_scan_1(&m, 0, 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(r.len(), 1);
        r.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(r.get(0).unwrap().0, 2);
    }
}

/*
    let mut last_d = -1.0;
    let mut r = Vec::new();
    let mut n = 0;

    for (d, idx) in v {
        if last_d < 0.0 {
            last_d = d;
        } else if d != last_d {
            n += 1;
            if n == k {
                break;
            }
        }
        r.push(idx);
    }
    r*/

