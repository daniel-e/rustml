extern crate num;

use std::fs::File;
use std::io::{Write, Result};
use std::process::{Command, Output};

static DEFAULT_OCTAVE_BIN: &'static str = "octave";

pub struct OctaveScriptBuilder {
    buf: Vec<String>,
    octave_bin: String
}

impl OctaveScriptBuilder {
    pub fn add(&self, s: &str) -> OctaveScriptBuilder {

        let mut buf = self.buf.clone();
        buf.push(s.to_string());
        OctaveScriptBuilder {
            buf: buf,
            octave_bin: self.octave_bin.clone()
        }
    }

    fn join(&self, v: &[f64]) -> String {

        let mut s = "[".to_string();

        for (idx, val) in v.iter().enumerate() {
            if idx > 0 {
                s = s + ",";
            }
            s = s + &format!("{}", val);
        }
        s + "]"
    }

    // TODO generic type
    pub fn add_values(&self, s: &str, vals: &[Vec<f64>]) -> OctaveScriptBuilder {

        let mut t = s.to_string();
        let n = vals[0].len();  // TODO error handling if vals is empty

        for i in (0..n) {
            let p = format!("${}", i + 1);
            let v = self.join(&vals.iter().map(|ref v| v[i]).collect::<Vec<f64>>());
            t = t.replace(&p, &v);
        }
        self.add(&t)
    }

    // TODO generic type
    pub fn add_vector(&self, s: &str, vals: &[f64]) -> OctaveScriptBuilder {

        let mut t = s.to_string();
        let v = self.join(vals);
        t = t.replace("$$", &v);
        self.add(&t)
    }

    // TODO generic type
    pub fn add_vals(&self, s: &str, vals: &[f64]) -> OctaveScriptBuilder {

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
