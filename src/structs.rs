pub struct Point2D<T> {
    pub x: T,
    pub y: T
}

impl <T> Point2D<T> {
    pub fn new(x: T, y: T) -> Point2D<T> {
        Point2D {
            x: x,
            y: y
        }
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

