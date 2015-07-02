//! Functions to read and write files (e.g. gzip compressed files).
extern crate flate2;
extern crate libc;

use std::fs::File;
use std::io::Read;
use std::io;
use self::flate2::read::GzDecoder;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

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
}
