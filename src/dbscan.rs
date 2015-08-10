//! Implementation of the DBSCAN clustering algorithm.

use std::iter;
use structs::Point2D;
use distance::DistancePoint2D;

pub fn dbscan(data: &Vec<Point2D<f64>>, eps: f64, minpts: usize) -> Vec<isize> {

    let mut s = ClusterDbscan::new(data, eps, minpts);
    s.compute()
}

struct ClusterDbscan<'a> {
    data: &'a Vec<Point2D<f64>>,
    eps: f64,
    minpts: usize,
    visited: Vec<bool>,
    cluster: Vec<isize>,
    c: isize
}

impl <'a> ClusterDbscan<'a> {
    pub fn new(data: &'a Vec<Point2D<f64>>, eps: f64, minpts: usize) -> ClusterDbscan {
        ClusterDbscan {
            data: data,
            eps: eps,
            minpts: minpts,
            visited: iter::repeat(false).take(data.len()).collect(),
            cluster: iter::repeat(-2).take(data.len()).collect(),
            c: -1
        }
    }

    fn visited(&self, pos: usize) -> bool {
        *self.visited.get(pos).unwrap()
    }

    fn visit(&mut self, pos: usize) {
        let visited = self.visited.get_mut(pos).unwrap();
        *visited = true;
    }

    fn neighbours(&self, pos: usize) -> Vec<usize> {
        let p = self.data.get(pos).unwrap();
        self.data.iter()
            .enumerate()
            .map(|(idx, q)| (idx, p.euclid(q)))
            .filter(|&(_idx, d)| d <= self.eps)
            .map(|(idx, _d)| idx)
            .collect()
    }

    fn noise(&mut self, pos: usize) {
        self.set_cluster(pos, -1);
    }

    pub fn compute(&mut self) -> Vec<isize> {

        for (idx, _p) in self.data.iter().enumerate() {
            if !self.visited(idx) {
                self.visit(idx);
                let neighbours = self.neighbours(idx);
                if neighbours.len() < self.minpts {
                    self.noise(idx);
                } else {
                    self.c += 1;
                    self.expand_cluster(idx, neighbours);
                }
            }
        }
        self.cluster.clone()
    }

    fn set_cluster(&mut self, pos: usize, c: isize) {
        let cl = self.cluster.get_mut(pos).unwrap();
        *cl = c;
    }

    fn get_cluster(&self, pos: usize) -> isize {
        *self.cluster.get(pos).unwrap()
    }

    fn expand_cluster(&mut self, ppos: usize, neighbours: Vec<usize>) {
        let c = self.c;
        self.set_cluster(ppos, c);
        
        let mut q = neighbours.clone();
        while q.len() > 0 {
            let pp = q.pop().unwrap();
            if !self.visited(pp) {
                self.visit(pp);
                let neighbours = self.neighbours(pp);
                if neighbours.len() >= self.minpts {
                    for n in neighbours {
                        q.push(n);
                    }
                }
            }
            if self.get_cluster(pp) == -2 {
                self.set_cluster(pp, c);
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use structs::Point2D;

    #[test]
    fn test_dbscan() {

        let data = vec![
            Point2D::new(1.0, 2.0),
            Point2D::new(5.0, 3.0),
            Point2D::new(2.0, 4.0),
            Point2D::new(3.0, 1.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(4.0, 3.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(2.0, 5.0),
            Point2D::new(14.0, 13.0),
            Point2D::new(11.0, 12.0),
            Point2D::new(12.0, 15.0),
            Point2D::new(32.0, 25.0),
        ];

        let r = dbscan(&data, 5.0, 3);
        assert_eq!(r, vec![0,0,0,0,0,0,0,0,1,1,1,-1]);
    }
}

