extern crate num;

use std::fmt;
use std::fs::File;
use std::io::{Write, Result};
use std::process::{Command, Output};

use matrix::Matrix;

static DEFAULT_OCTAVE_BIN: &'static str = "octave";

pub struct OctaveScriptBuilder {
    buf: Vec<String>,
    octave_bin: String
}

impl OctaveScriptBuilder {
    /// Adds the string to the Octave script.
    ///
    /// At the end of the line a semicolon is appended.
    pub fn add(&self, s: &str) -> OctaveScriptBuilder {

        let mut buf = self.buf.clone();
        buf.push(s.to_string());
        OctaveScriptBuilder {
            buf: buf,
            octave_bin: self.octave_bin.clone()
        }
    }

    fn join<T: fmt::Display>(&self, v: &[T]) -> String {

        let mut s = "[".to_string();

        for (idx, val) in v.iter().enumerate() {
            if idx > 0 {
                s = s + ",";
            }
            s = s + &format!("{}", val);
        }
        s + "]"
    }

    /// Adds the string to the Octave script.
    ///
    /// At the end of the line a semicolon is appended.
    /// 
    /// If the string contains a dollar sign followed by a number `i` (e.g. `$1` or `$12`)
    /// this placeholder will be replaced by the column if matrix `m` at column
    /// `i-1` (i.e. $1 is replaced by the first column of `m`).
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::octave::*;
    /// use rustml::matrix::Matrix;
    ///
    /// # pub fn main() {
    /// let m = mat![
    ///     1, 2, 3;
    ///     4, 5, 6
    /// ];
    /// let s = builder().add_columns("x = $1; y = $2", &m);
    /// assert_eq!(
    ///     s.to_string(),
    ///     "1;\nx = [1,4]; y = [2,5];\n"
    /// );
    /// # }
    /// ```
    pub fn add_columns<T: fmt::Display + Copy>(&self, s: &str, m: &Matrix<T>) -> OctaveScriptBuilder {

        let mut t = s.to_string();
        let n = m.cols();

        for i in (0..n) {
            let p = format!("${}", i + 1);
            let v = self.join(&m.row_iter().map(|ref v| v[i]).collect::<Vec<T>>());
            t = t.replace(&p, &v);
        }
        self.add(&t)
    }

    /// Adds the string to the Octave script.
    ///
    /// At the end of the line a semicolon is appended. If the string contains two
    /// consecutive dollar signed (i.e. `$$` these will be replaced by a vector
    /// containing the elements of `vals`.
    /// 
    /// # Example
    ///
    /// ```
    /// # extern crate rustml;
    /// use rustml::octave::*;
    ///
    /// # pub fn main() {
    /// let s = builder().add_vector("x = $$", &[1, 2, 3]);
    /// assert_eq!(
    ///     s.to_string(),
    ///     "1;\nx = [1,2,3];\n"
    /// );
    /// # }
    /// ```
    pub fn add_vector<T: fmt::Display>(&self, s: &str, vals: &[T]) -> OctaveScriptBuilder {

        let mut t = s.to_string();
        let v = self.join(vals);
        t = t.replace("$$", &v);
        self.add(&t)
    }

    /// Adds the string to the Octave script.
    ///
    /// At the end of the line a semicolon is appended.
    ///
    /// If the string contains a dollar sign followed by a number `i` (e.g. `$1` or `$12`)
    /// this placeholder will be replaced by the value that is stored in the vector
    /// `vals` at index `i-1`.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate rustml;
    /// use rustml::octave::*;
    ///
    /// # pub fn main() {
    /// let s = builder().add_values("x = $1 + $2", &[5, 3]);
    /// assert_eq!(
    ///     s.to_string(),
    ///     "1;\nx = 5 + 3;\n"
    /// );
    /// # }
    /// ```
    pub fn add_values<T: fmt::Display>(&self, s: &str, vals: &[T]) -> OctaveScriptBuilder {

        let mut t = s.to_string();
        let n = vals.len();

        for i in (0..n) {
            let p = format!("${}", i + 1);
            let v = format!("{}", vals[i]);
            t = t.replace(&p, &v);
        }
        self.add(&t)
    }

    pub fn octave_bin(&self, path: &str) -> OctaveScriptBuilder {
        OctaveScriptBuilder {
            buf: self.buf.clone(),
            octave_bin: path.to_string()
        }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        s = s + "1;\n";
        for j in &self.buf {
            s = s + &j + ";\n";
        }
        s
    }

    pub fn write(&self, filename: &str) -> Result<()> {

        match File::create(filename) {
            Ok(mut f) => {
                let data = self.to_string().into_bytes();
                f.write_all(&data)
            },
            Err(e) => Err(e)
        }
    }

    pub fn run(&self, filename: &str) -> Result<Output> {

        match self.write(filename) {
            Ok(_) => {
                let mut c = self.octave_bin.clone();
                c = c + " " + filename;

                Command::new("sh")
                    .arg("-c")
                    .arg(c)
                    .output()
            }
            Err(e) => Err(e)
        }
    }
}

pub fn builder() -> OctaveScriptBuilder {
    OctaveScriptBuilder {
        buf: vec![],
        octave_bin: DEFAULT_OCTAVE_BIN.to_string()
    }
}
