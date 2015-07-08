//! Functions to parse and create comma-separated values (CSV).
use std::fs::File;
use std::io::Read;
use std::f64;
use std::str::FromStr;
use std::fmt;

use matrix::*;

/// Converts a vector into a comma seperated list of values.
///
/// As delimiter the string `sep` is used.
///
/// # Example
///
/// ```
/// use rustml::csv::vec_to_csv;
///
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
/// # #[macro_use] extern crate rustml;
/// use rustml::*;
/// use rustml::csv::matrix_to_csv;
///
/// # fn main() {
/// let m = mat![1.0, 2.0; 3.0, 4.0];
/// assert_eq!(matrix_to_csv(&m, ","), "1,2\n3,4\n")
/// # }
/// ```
pub fn matrix_to_csv<T: fmt::Display + Clone>(m: &Matrix<T>, sep: &str) -> String {

    m.row_iter()
        .map(|row| vec_to_csv(row, sep))
        .fold(String::new(), |s, val| s + &val + "\n")
}


/// Reads a matrix from a CSV.
///
/// The elements are of the type f64.
pub fn from_csv_string(s: &str, sep: &str) -> Result<Matrix<f64>, &'static str> {

    // TODO
    let v = s.split('\n')
        .map(|x| match x.find('#') { // remove comments
            Some(pos) => {
                let mut tmp = x.to_string();
                tmp.truncate(pos);
                tmp
            }
            _ => x.to_string()
        })
        .filter(|x| x.trim().len() > 0)
        .map(|x| x.split(sep)     // split each line by the given separator
            .map(|x| x.trim())
            .map(|x| f64::from_str(x).unwrap_or(f64::NAN))
            .collect::<Vec<f64>>()
        )
        .collect::<Vec<Vec<f64>>>();

    let rows = v.len();
    let cols = v.first().unwrap_or(&Vec::new()).len();
    let data = v.iter().flat_map(|x| x.iter()).cloned().collect::<Vec<f64>>();

    // now we have the data two times in memory
    // performance

    if rows == 0 
        || cols == 0
        || v.iter().any(|x| x.len() != cols)
        || data.iter().any(|x| x.is_nan()) {
        return Err("Invalid format.");
    }

    match Matrix::from_vec(data, rows, cols) {
        Some(m) => Ok(m),
        _ => Err("Could not create matrix.")
    }
}

pub fn from_csv_file(fname: &str, sep: &str) -> Result<Matrix<f64>, &'static str> {

    let mut s = String::new();

    try!(
        try!(
            File::open(fname).map_err(|_| "Could not open file")
        )
        .read_to_string(&mut s).map_err(|_| "Could not read file.")
    );
    from_csv_string(&s, sep)
}


#[cfg(test)]
mod tests {
    use super::*;
    use matrix::*;

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

    #[test]
    fn test_from_csv_string() {

        let mut s = "1,2,3\n4,5,6";
        let mut m = from_csv_string(s, ",").unwrap();;
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.row(0).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(m.row(1).unwrap(), [4.0, 5.0, 6.0]);

        s = "";
        assert!(from_csv_string(s, ",").is_err());

        s = "1";
        m = from_csv_string(s, ",").unwrap();
        assert_eq!(m.cols(), 1);
        assert_eq!(m.rows(), 1);

        s = "1,2\n,3";
        assert!(from_csv_string(s, ",").is_err());

        s = "#abc";
        assert!(from_csv_string(s, ",").is_err());

        s = "#aaa\n1,2,3#bla\n3,4,4#abc\n\n#hhhh";
        m = from_csv_string(s, ",").unwrap();
        assert_eq!(m.cols(), 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.row(0).unwrap(), [1.0, 2.0, 3.0]);
        assert_eq!(m.row(1).unwrap(), [3.0, 4.0, 4.0]);
    }
}

