//! Functions to read and write files (e.g. gzip compressed files, csv files, etc).
extern crate flate2;
extern crate libc;
extern crate regex;

use std::fs::File;
use std::io::{Read, BufReader, BufRead, Stdin, stdin};
use self::flate2::read::GzDecoder;
use self::regex::Regex;
use std::iter::Skip;
use std::slice::Iter;
use std::fmt;
use std::str::FromStr;
use std::{io as stdio};

use vectors::copy_memory;
use matrix::Matrix;

// ----------------------------------------------------------------------------

/// Create comman separated values from a collection.
pub trait CsvString {

    /// Converts a data structure into a comma seperated list of values.
    ///
    /// As delimiter the string `sep` is used.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::io::CsvString;
    ///
    /// let a = [1, 2, 3];
    /// assert_eq!(a.to_csv(","), "1,2,3");
    /// ```
    fn to_csv(&self, delim: &str) -> String;
}

impl <T: fmt::Display> CsvString for Vec<T> {

    fn to_csv(&self, delim: &str) -> String {

        self[..].to_csv(delim)
    }
}

impl <T: fmt::Display> CsvString for [T] {

    fn to_csv(&self, delim: &str) -> String {

        self.iter().enumerate()
            .map(|(c, val)| 
                match c {
                    0 => format!("{}", val),
                    _ => format!("{}{}", delim, val)
                }
            )
            .fold(String::new(), |s, val| s + &val)
    }
}

impl <T: fmt::Display + Clone> CsvString for Matrix<T> {

    fn to_csv(&self, delim: &str) -> String {

        self.row_iter()
            .map(|row| row.to_csv(delim))
            .fold(String::new(), |s, val| s + &val + "\n")
    }
}

// ----------------------------------------------------------------------------

/// Struct to decompress gzip streams.
pub struct GzipData {
    v: Vec<u8>,
    idx: usize
}

impl <'b> GzipData {

    /// Reads gzip data from a file and returns the uncompressed data
    /// in a vector. Returns an error message on failure.
    pub fn from_file(fname: &str) -> Result<GzipData, &'static str> {

        let mut r: Vec<u8> = Vec::new();
        try!(
            try!(GzDecoder::new(
                    try!(File::open(fname)
                         .map_err(|_| "Could not open file")
                    )
                )
                .map_err(|_| "Invalid gzip header.")
            )
            .read_to_end(&mut r)
                .map_err(|_| "Could not unzip data.")
        );
        Ok(GzipData {
            v: r,
            idx: 0
        })
    }

    // TODO test
    pub fn from_buf(v: Vec<u8>) -> GzipData {

        GzipData {
            v: v,
            idx: 0
        }
    }

    /// Returns the uncompressed data.
    pub fn into_bytes(&self) -> Vec<u8> { self.v.clone() }

    /// Returns the length of the uncomressed data.
    pub fn len(&self) -> usize { self.v.len() }

    /// Returns an iterator over the uncompressed data.
    pub fn iter(&self) -> Skip<Iter<u8>> { self.v.iter().skip(self.idx) }
    // TODO test

    // TODO test
    pub fn buf(&'b self) -> &'b [u8] { &self.v.split_at(self.idx).1 }
}

/// Implementation of the `Read` trait for GzipData.
impl Read for GzipData {

    fn read(&mut self, buf: &mut [u8]) -> stdio::Result<usize> {

        if self.idx >= self.v.len() {
            return Ok(0);
        }

        let n = buf.len();
        let c = copy_memory(
            buf, 
            self.v.split_at(self.idx).1, 
            n
        );
        self.idx += c;
        Ok(c)
    }
}

// -------------------------------------------------------------------------

/// Read lines from a reader and returns the line if it matches with
/// a regex.
///
/// The call to the method next returns `None` if the function
/// `read_line` of the buffer from which the lines are read returns
/// `Ok(0)`. Otherwise a result is returned in a `Some`. The result
/// is an error if the buffer's `read_line` returns an error.
/// Otherwise the result contains a vector of captures as strings.
/// The first element of the vector contains the whole match.
///
/// See the functions [match_lines](fn.match_lines.html) and
/// [match_lines_stdin](fn.match_lines_stdin.html) for details.
pub struct MatchLines<R: Read> {
    reader: BufReader<R>,
    r: Regex,
}

impl <R: Read> Iterator for MatchLines<R> {
    type Item = stdio::Result<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut buf = String::new();
            match self.reader.read_line(&mut buf) {
                Ok(0) => {
                    return None
                },
                Err(e) => {
                    return Some(Err(e))
                },
                Ok(_n) => {
                    // remove new line from end of line
                    if buf.ends_with("\n") {
                        buf.pop();
                    }
                    match self.r.captures(&buf) {
                        Some(cap) => {
                            return Some(Ok(cap.iter().map(|s| s.unwrap().to_string()).collect()));
                        },
                        _ => ()  // skip line and try next line
                    }
                }
            }
        }
    }

}

/// Returns an instance of `MatchLines` which can be used to read all
/// lines from `stdin` and match each line with the provided regex. Only
/// those lines are returned which match the regex.
///
/// # Example
///
/// ```ignore
/// #[macro_use] extern crate rustml;
/// extern crate regex;
///
/// use regex::Regex;
/// use rustml::io::match_lines;
///
/// # fn main() {
/// let r = Regex::new(r"^[a-z]+ (\d+)$").unwrap();
/// for line in match_lines(r) {
///     let captures = line.unwrap();
///     println!("{}", captures[1]);
/// }
/// # }
/// ```
pub fn match_lines_stdin(r: Regex) -> MatchLines<Stdin> {
    MatchLines {
        reader: BufReader::new(stdin()),
        r: r,
    }
}

/// Returns an instance of `MatchLines` which can be used to read all
/// lines from the given reader. Each line is matched against the provided regex
/// and only those lines which match the regex are returned.
///
/// # Example
///
/// ```
/// #[macro_use] extern crate rustml;
/// extern crate regex;
///
/// use std::fs::File;
/// use std::io::BufReader;
/// use regex::Regex;
/// use rustml::io::match_lines;
///
/// # fn main() {
/// // the file lines.txt contains the lines:
/// // line 1
/// // line 2
/// // line 3
/// let f = File::open("datasets/testing/lines.txt").unwrap();
/// let r = BufReader::new(f);
/// let mut v = Vec::new();
/// for line in match_lines(r, Regex::new(r"^[a-z]+ (\d+)$").unwrap()) {
///     let captures = line.unwrap();
///     v.push(captures[1].parse::<usize>().unwrap());
/// }
/// assert_eq!(v, vec![1, 2, 3]);
/// # }
/// ```
pub fn match_lines<R: Read>(reader: R, r: Regex) -> MatchLines<R> {
    MatchLines {
        reader: BufReader::new(reader),
        r: r
    }
}

// -------------------------------------------------------------------------

/// Convert a collection into a format that can be read with Octave.
pub trait OctaveString {
    fn to_octave(&self, name: &str) -> String;
}

impl <T: fmt::Display + Clone> OctaveString for Matrix<T> {

    fn to_octave(&self, name: &str) -> String {
        format!(
            "# name: {}\n# type: matrix\n# rows: {}\n# columns: {}\n{}",
            name, self.rows(), self.cols(), self.to_csv(" ")
        )
    }
}

impl <T: fmt::Display> OctaveString for Vec<T> {

    fn to_octave(&self, name: &str) -> String {
        format!(
            "# name: {}\n# type: matrix\n# rows: {}\n# columns: {}\n{}\n",
            name, self.len(), 1, self.to_csv("\n")
        )
    }
}

// -------------------------------------------------------------------------

/// Iterator to read comma separated values from a reader.
/// 
/// The default delimiter is `,`. It can be changed via the method 
/// [delimiter](#method.delimiter). For each line that is read via
/// the reader `R` a `Result` is returned which contains an error if
/// an error occurred or which contains a vector of strings 
/// representing the values.
///
/// Comments are removed and empty lines will be skipped.
///
/// # Examples
///
/// Read CSV file via an iterator.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use std::io::{BufRead, Cursor, Read, BufReader};
/// use rustml::io::*;
/// use rustml::matrix::Matrix;
///
/// # fn main() {
/// let s = "1,2,3\n4,5,6";
/// let r = BufReader::new(Cursor::new(s.as_bytes()));
/// let mut i = csv_reader(r);
///
/// assert_eq!(i.next().unwrap().unwrap(), vec!["1", "2", "3"]);
/// assert_eq!(i.next().unwrap().unwrap(), vec!["4", "5", "6"]);
/// assert!(i.next().is_none());
/// # }
/// ```
///
/// Create a matrix from a CSV file.
/// 
/// ```
/// # #[macro_use] extern crate rustml;
/// use std::io::{BufRead, Cursor, Read, BufReader};
/// use rustml::io::*;
/// use rustml::matrix::Matrix;
///
/// # fn main() {
/// let s = "1,2,3\n4,5,6";
/// let r = BufReader::new(Cursor::new(s.as_bytes()));
/// let csv = csv_reader(r);
/// let m = Matrix::<usize>::from_csv(csv).unwrap();
/// assert_eq!(m, mat![1,2,3; 4,5,6]);
/// # }
/// ```
pub struct CsvReader<R: Read> {
    reader: BufReader<R>,
    delim: String
}

impl <R: Read> CsvReader<R> {

    pub fn delimiter(self, delim: &str) -> CsvReader<R> {
        CsvReader {
            reader: self.reader,
            delim: delim.to_string()
        }
    }
}

impl <R: Read> Iterator for CsvReader<R> {

    type Item = stdio::Result<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut buf = String::new();
            match self.reader.read_line(&mut buf) {
                Ok(0) => {
                    return None
                },
                Err(e) => {
                    return Some(Err(e))
                },
                Ok(_n) => {
                    // remove comments
                    let nc = match buf.find('#') {
                        Some(pos) => {
                            let mut tmp = buf.clone();
                            tmp.truncate(pos);
                            tmp
                        },
                        _ => buf
                    };

                    if nc.trim().len() > 0 {
                        return Some(Ok(
                            nc.split(&self.delim)
                                .map(|x| x.trim().to_string()) .collect::<Vec<String>>()
                        ));
                    }
                }
            }
        }
    }
}

pub fn csv_reader<R: Read>(reader: R) -> CsvReader<R> {
    CsvReader {
        reader: BufReader::new(reader),
        delim: ",".to_string()
    }
}

// -------------------------------------------------------------------------

/// Trait to build a collection from comma separated values.
pub trait FromCsv: Sized {
    /// Reads comma separated values via a reader.
    ///
    /// Returns a result which contains the collection on success or
    /// a string with an error message on failure.
    fn from_csv<R: Read>(reader: CsvReader<R>) -> Result<Self, String>;
}

impl <T: FromStr + Clone> FromCsv for Matrix<T> {

    fn from_csv<R: Read>(reader: CsvReader<R>) -> Result<Matrix<T>, String> {
        
        let mut m = Matrix::new();
        for i in reader {
            let mut v = Vec::new();
            for j in i.unwrap() {
                match j.parse::<T>() {
                    Ok(val) => v.push(val),
                    _ => {
                        return Err(format!("Could not parse the value: {}", j));
                    }
                }
            }
            m.add_row(&v);
        }
        Ok(m)
    }
}

// TODO impl for Vec<T>

// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate regex;

    use super::*;
    use std::io::Read;
    use self::regex::Regex;
    use std::fs::File;
    use std::io::BufReader;
    use matrix::Matrix;

    #[test]
    fn test_read_gzip() {

        assert_eq!(
            String::from_utf8(
                GzipData::from_file("datasets/testing/hello_world.gz").unwrap().into_bytes()).unwrap(), 
            "hello world".to_string()
        );

        assert_eq!(GzipData::from_file("datasets/testing/hello_world.gz").unwrap().len(), 11);
        assert!(GzipData::from_file("datasets/testing/random.data").is_err());

        let mut data = GzipData::from_file("datasets/testing/hello_world.gz").unwrap();
        let mut v: Vec<u8> = Vec::new();
        assert!(data.read_to_end(&mut v).is_ok());
        assert_eq!(String::from_utf8(v).unwrap(), "hello world".to_string());
    }

    #[test]
    fn test_match_lines() {

        let f = File::open("datasets/testing/lines.txt").unwrap();
        let r = BufReader::new(f);
        let mut v = Vec::new();
        for line in match_lines(r, Regex::new(r"^[a-z]+ (\d+)$").unwrap()) {
            let captures = line.unwrap();
            v.push(captures[1].parse::<usize>().unwrap());
        }
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_matrix_to_octave_string() {

        let m = mat![1, 2, 3; 4, 5, 6];
        let s = m.to_octave("mymatrix");
        assert_eq!(s,
            "# name: mymatrix\n# type: matrix\n# rows: 2\n# columns: 3\n1 2 3\n4 5 6\n"
        );
    }

    #[test]
    fn test_vec_to_octave_string() {

        let m = vec![1,2,3,4];
        let s = m.to_octave("myvec");
        assert_eq!(s,
            "# name: myvec\n# type: matrix\n# rows: 4\n# columns: 1\n1\n2\n3\n4\n"
        );
    }

    #[test]
    fn test_vec_to_csv() {

        let v = vec![1, 2, 3, 4];
        assert_eq!(v.to_csv(","), "1,2,3,4");

        let s = [1, 2, 3];
        assert_eq!(s.to_csv(","), "1,2,3");

        let a = [1];
        assert_eq!(a.to_csv(","), "1");

        let b = [1,2];
        assert_eq!(b.to_csv(","), "1,2");

        let c = Vec::<usize>::new();
        assert_eq!(c.to_csv(","), "");
    }

    #[test]
    fn test_mat_to_csv() {

        let m = mat![1, 2, 3; 4, 5, 6];
        assert_eq!(m.to_csv(","), "1,2,3\n4,5,6\n");
    }

    #[test]
    fn test_csv_reader() {

        let f = File::open("datasets/testing/csv.txt").unwrap();
        let r = csv_reader(f);
        let v = r.map(|x| x.unwrap()).collect::<Vec<Vec<String>>>();;
        
        assert_eq!(v, vec![
            vec!["1","2","3","4"],
            vec!["5","6","7","8"],
            vec!["9","10","11","12"]
        ]);
    }

    #[test]
    fn test_csv_reader_matrix() {
        let f = File::open("datasets/testing/csv.txt").unwrap();
        let r = csv_reader(f);
        let m = Matrix::<usize>::from_csv(r).unwrap();
        assert_eq!(m, mat![1,2,3,4; 5,6,7,8; 9,10,11,12]);
    }
}
