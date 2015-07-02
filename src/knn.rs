extern crate num;

use self::num::traits::Float;
use matrix::*;

/// Search the k nearest neighbours for the given example.
pub fn knn_scan<D, T: Float>(m: &Matrix<T>, example: &[T], k: usize, df: D) -> Option<Vec<usize>>
    where D : Fn(&[T], &[T]) -> T {

    if example.len() != m.cols() {
        return None;
    }

    let mut near: Vec<(usize, T)> = Vec::with_capacity(k);

    for (idx, row) in m.row_iter().enumerate() {
        let d = df(row, example);

        // search the first neighbour for which the distance is larger
        // than the distance to the current example and insert it at that
        // position
        let p = near.iter().position(|&(_, val)| val > d);
        match p {
            Some(pos) => {
                near.insert(pos, (idx, d));
                if near.len() > k {
                    near.pop();
                }
            }
            _ => {
                if idx < k {
                    near.push((idx, d))
                }
            }
        }
    }

    Some(near.iter().map(|&(idx, _)| idx.clone()).collect())
}


#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;
    use distance::*;

    #[test]
    fn test_knn_scan() {

        let mut m = mat![
            1.0, 2.0;
            2.0, 2.0;
            3.0, 3.0
        ];

        let a = knn_scan(&m, &[1.0, 1.0, 2.0], 1, |x, y| Euclid::compute(x, y).unwrap());
        assert!(a.is_none());

        let mut label = knn_scan(&m, &[1.0, 1.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![0]);

        label = knn_scan(&m, &[1.0, 2.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![0]);

        label = knn_scan(&m, &[2.0, 2.2], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![1]);

        label = knn_scan(&m, &[5.0, 6.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![2]);

        // ---------------

        m = mat![
            1.0, 2.0;
            1.3, 1.8;
            1.2, 2.1;
            2.0, 2.0
        ];

        label = knn_scan(&m, &[1.1, 2.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![0]);
        label = knn_scan(&m, &[1.1, 2.0], 2, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![0, 2]);
        label = knn_scan(&m, &[1.1, 2.0], 3, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![0, 2, 1]);
        label = knn_scan(&m, &[1.1, 2.0], 4, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, vec![0, 2, 1, 3]);
    }
}


