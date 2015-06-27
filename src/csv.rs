use std::fmt;
use ::matrix::*;

/// Converts a vector into a comma seperated list of values.
///
/// As delimiter the string `sep` is used.
///
/// # Example
///
/// ```
/// let a = &[1, 2, 3];
/// assert_eq!(vec_to_csv(a, ","), "1,2,3");
/// ```
pub fn vec_to_csv<T: fmt::Display>(v: &[T], sep: &str) -> String {

    v.iter().enumerate()
        .map(|(c, val)| 
            match c {
                0 => format!("{}", val),
                _ => format!("{}{}", sep, val)
            }
        )
        .fold(String::new(), |s, val| s + &val)
}

/// Converts a matrix into a comma seperated list of values.
///
/// As delimiter between the columns the string `sep` is used. As
/// delimiter between the rows a newline is used.
///
/// # Example
///
/// ```
/// let m = mat![1.0, 2.0; 3.0, 4.0];
/// assert_eq!(matrix_to_csv(&m, ","), "1,2\n3,4\n")
/// ```
pub fn matrix_to_csv<T: fmt::Display + Clone>(m: &Matrix<T>, sep: &str) -> String {

    m.row_iter()
        .map(|row| vec_to_csv(row, sep))
        .fold(String::new(), |s, val| s + &val + "\n")
}


#[cfg(test)]
mod tests {
    use super::*;
    use ::matrix::*;

    #[test]
    fn test_vec_to_csv() {

        let b = &[1, 2, 3];
        assert_eq!(vec_to_csv(b, ","), "1,2,3");

        let mut a = vec![1, 2, 3];
        assert_eq!(vec_to_csv(&a, ","), "1,2,3");
        a = vec![1];
        assert_eq!(vec_to_csv(&a, ","), "1");
        a = vec![];
        assert_eq!(vec_to_csv(&a, ","), "");
    }

    #[test]
    fn test_matrix_to_csv() {

        let m = mat![1.0, 2.0; 3.0, 4.0];
        assert_eq!(matrix_to_csv(&m, ","), "1,2\n3,4\n")
    }
}

