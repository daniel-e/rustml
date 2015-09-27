use std::process::Command;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::Write;


fn try_gcc(lib: &str, msg: &str) {

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("main.c");
    let mut f = File::create(&dest_path).unwrap();

    f.write_all(b"
        int main() {
        }
    ").unwrap();

    let s = Command::new("gcc")
        .args(&[dest_path.into_os_string().to_str().unwrap(), lib, "-o"])
        .arg(&format!("{}/main.o", out_dir))
        .status()
        .unwrap();

    assert!(s.success(), "\n\n".to_string() + msg);
}

fn main() {
    try_gcc("-lblas", "BLAS not found. On Ubuntu try 'sudo apt-get install libblas3' before continuing.");
    try_gcc("-lopencv_highgui", "OpenCV not found. On Ubuntu try 'sudo apt-get install libopencv-highgui2.4' before continuing.");
    try_gcc("-lopencv_corex", "OpenCV not found. On Ubuntu try 'sudo apt-get install libopencv-core2.4' before continuing.");
    try_gcc("-lopencv_imgproc", "OpenCV not found. On Ubuntu try 'sudo apt-get install libopencv-imgproc2.4' before continuing.");
}

