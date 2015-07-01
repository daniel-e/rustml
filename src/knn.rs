
use std::cmp::Ordering;
use ::matrix::*;
use ::vectors::group;

pub fn knn_scan<D>(m: &Matrix<f64>, example: &[f64], k: usize, df: D) -> Option<f64>
    where D : Fn(&[f64], &[f64]) -> f64 {

    let mut labels: Vec<(f64, f64)> = Vec::new();

    for row in m.row_iter() {
        let (l, ex) = row.split_at(1);
        let label = l.get(0).unwrap().clone();
        let d = df(ex, example);

        // search the first neighbour for which the distance is larger
        // than the distance to the current example and insert it at that
        // position
        let p = labels.iter().position(|&(_, val)| val > d);
        match p {
            Some(pos) => labels.insert(pos, (label, d)),
            _         => labels.push((label, d))
        }
        labels.truncate(k);
    }

    // get the labels of the k nearest neighbours
    let mut l: Vec<f64> = labels.iter().map(|&(label, _)| label.clone()).collect();

    // sort the labels
    l.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));
    // group the labels and count them
    let mut r = group(&l);
    // find the correct label
    r.sort_by(|a, b| a.1.cmp(&b.1));

    match r.last() {
        None => None,
        Some(x) => Some(x.0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ::matrix::*;
    use ::distance::*;

    #[test]
    fn test_knn_scan() {

        let mut m = mat![
            10.0, 1.0, 2.0;
            11.0, 2.0, 2.0;
            12.0, 3.0, 3.0
        ];

        let mut label = knn_scan(&m, &[1.0, 1.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 10.0);

        label = knn_scan(&m, &[1.0, 2.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 10.0);

        label = knn_scan(&m, &[2.0, 2.2], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 11.0);

        label = knn_scan(&m, &[5.0, 6.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 12.0);

        // ---------------

        m = mat![
            10.0, 1.0, 2.0;
            11.0, 1.3, 1.8;
            10.0, 1.2, 2.1;
            12.0, 2.0, 2.0
        ];

        label = knn_scan(&m, &[1.1, 2.0], 1, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 10.0);
        label = knn_scan(&m, &[1.1, 2.0], 2, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 10.0);
        label = knn_scan(&m, &[1.1, 2.0], 3, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 10.0);
        label = knn_scan(&m, &[1.1, 2.0], 4, |x, y| Euclid::compute(x, y).unwrap()).unwrap();
        assert_eq!(label, 10.0);
    }
}


