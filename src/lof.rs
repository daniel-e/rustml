use super::matrix::*;

pub struct Lof {
    p: Matrix<f64>,
}

impl Lof {

    pub fn new() -> Lof {
        Lof {
            p: Matrix::fill(0.0, 1, 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Lof;
    use ::matrix::*;

    fn distances(m: &Matrix<f64>) -> Matrix<f64> {

        let mut r: Matrix<f64> = Matrix::fill(0.0, m.rows(), m.rows());


        r
    }

    #[test]
    pub fn test_lof() {

        let m: Matrix<f64> = mat![0.1, 0.1; 0.2, 0.2; 0.2, 0.2; 0.3, 0.3];
        let k = distances(&m);
        println!("{}", m);
        println!("{}", k);
    }
}

