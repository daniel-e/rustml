
use std::cmp::Ordering;
use std::collections::BTreeMap;
use ::matrix::*;

// TODO refactoring

/// Searches for the k nearest neighbours for a given row in all rows
/// (except the one indexed by idx) of the given matrix.
///
/// If m is the number of rows and n is the number of columns the
/// complexity is O(mn + m log m).
pub fn knn_scan<D>(m: &Matrix<f64>, idx: usize, k: usize, df: D) -> Option<Vec<(usize, f64)>> 
    where D : Fn(&[f64], &[f64]) -> f64 {

    // TODO http://doc.rust-lang.org/std/cmp/trait.PartialOrd.html
    // NaN

    let row = m.row(idx).expect("Could not fetch row.");

    let mut v: Vec<(usize, f64)> = m.row_iter().enumerate() // O(m)
        .filter(|&(i, _)| i != idx)
        .map(|(i, r)| (i, df(r, row))) // O(n)
        .collect();

    //O(m log m)
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    Some(v.iter().take(k).cloned().collect())
}

// TODO move into another module
pub fn group(v: &Vec<f64>) -> Vec<(f64, usize)> {

    let mut r: Vec<(f64, usize)> = Vec::new();
    for val in v {
        if r.len() == 0 {
            r.push((*val, 1));
        } else {
            let mut x = r.pop().unwrap();
            if x.0 != *val {
                r.push(x);
                x = (*val, 0);
            }
            x.1 += 1;
            r.push(x);
        }
    }
    r
}

pub fn knn_search<D>(m: &Matrix<f64>, example: &[f64], k: usize, df: D) -> Option<f64>
    where D : Fn(&[f64], &[f64]) -> f64 {

    let mut labels: Vec<(f64, f64)> = Vec::new();

    for row in m.row_iter() {
        let (l, ex) = row.split_at(1);
        let label = l.get(0).unwrap().clone();
        let d = df(ex, example);

        if labels.len() == 0 {
            labels.push((label, d));
        } else {
            let mut pos = k;
            for (idx, val) in labels.iter().enumerate() {
                if d < val.1 {
                    pos = idx;
                    break;
                }
            }
            if pos < k {
                labels.insert(pos, (label, d));
                labels.truncate(k);
            }
        }
    }

    let mut l: Vec<f64> = labels.iter().map(|&(label, _)| label.clone()).collect();

    l.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));
    let mut r = group(&l);
    // TODO
    Some(0.0) 
}


#[cfg(test)]
mod tests {
    use super::*;
    use ::matrix::*;
    use ::distance::*;

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
    fn test_knn_scan() {

        let m = mat![1.0, 2.0; 3.0, 3.0; 1.0, 2.0; 5.0, 6.0; 10.0, 12.0];

        // 1st nearest neighbours of [3.0, 3.0]
        let mut r = knn_scan(&m, 1, 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(r.len(), 1);
        r.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(r.get(0).unwrap().0, 0);

        // 2-nearest neighbours of [3.0, 3.0] (= [1.0, 2.0], [1.0, 2.0])
        r = knn_scan(&m, 1, 2, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(r.len(), 2);
        r.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(r.get(0).unwrap().0, 0);
        assert_eq!(r.get(1).unwrap().0, 2);

        // 1-nearest neighbour of [1.0, 2.0] (= [1.0, 2.0])
        r = knn_scan(&m, 0, 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
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

