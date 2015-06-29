extern crate flate2;

use std::fs::File;
use std::io::Read;
use self::flate2::read::GzDecoder;

/// Reads gzip data from a file and returns the uncompressed data
/// in a vector. Returns an error message on failure.
pub fn read_gzip(fname: &str) -> Result<Vec<u8>, &'static str> {

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
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_gzip() {

        assert_eq!(
            String::from_utf8(read_gzip("datasets/testing/hello_world.gz").unwrap()).unwrap(), 
            "hello world".to_string()
        );

        assert!(read_gzip("datasets/testing/random.data").is_err());
    }
}
