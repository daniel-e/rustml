//! Functions to read and write files (e.g. gzip compressed files).
extern crate flate2;
extern crate libc;
extern crate regex;

use std::fs::File;
use std::io::{Read, BufReader, BufRead, Stdin, stdin};
use std::io;
use self::flate2::read::GzDecoder;
use self::regex::Regex;
use std::iter::Skip;
use std::slice::Iter;

use vectors::copy_memory;

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

    /// Returns an iterator over the uncompressed data. TODO test
    pub fn iter(&self) -> Skip<Iter<u8>> { self.v.iter().skip(self.idx) }

    // TODO test
    pub fn buf(&'b self) -> &'b [u8] { &self.v.split_at(self.idx).1 }
}

/// Implementation of the `Read` trait for GzipData.
impl Read for GzipData {

    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {

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
    type Item = io::Result<Vec<String>>;

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

#[cfg(test)]
mod tests {
    extern crate regex;

    use super::*;
    use std::io::Read;
    use self::regex::Regex;
    use std::fs::File;
    use std::io::BufReader;

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

}
