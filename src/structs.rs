//! Collection of some common data structures.
use std::fmt;

// A point with two dimensions, `x` and `y`.
pub struct Point2D<T> {
    pub x: T,
    pub y: T
}

impl <T> Point2D<T> {
    // Creates a new point with two dimensions.
    pub fn new(x: T, y: T) -> Point2D<T> {
        Point2D {
            x: x,
            y: y
        }
    }
}

impl <T: fmt::Display + Clone> fmt::Display for Point2D<T> {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} {})", self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d() {
        let p = Point2D::new(2, 3);
        assert_eq!(p.x, 2);
        assert_eq!(p.y, 3);
    }
}

