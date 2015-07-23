//! Bindings for OpenCV.

extern crate libc;

use self::libc::{c_char, c_int, c_void};

#[repr(C)]
pub struct CvCapture;

// http://docs.opencv.org/modules/core/doc/old_basic_structures.html
#[repr(C)]
pub struct IplImage {
    nsize: c_int,
    id: c_int,
    pub nchannels: c_int,
    alphachannel: c_int,
    depth: c_int,
    colormodel: [c_char; 4],
    channelseq: [c_char; 4],
    dataorder: c_int,
    origin: c_int,
    align: c_int,
    pub width: c_int,
    pub height: c_int,
    roi: *mut c_void, // actually it is not a void pointer
    maskroi: *mut c_void, // actually it is not a void pointer
    imageid: *mut c_void,
    titleinfo: *mut c_void,
    imagesize: c_int,
    pub imagedata: *mut c_char,
    pub widthstep: c_int,
    bordermode: [c_int; 4],
    borderconst: [c_int; 4],
    imagedataorigin: *mut c_char
}

#[link(name = "opencv_highgui")]
extern {
    pub fn cvCreateFileCapture(fname: *const c_char) -> *mut CvCapture;

    pub fn cvGrabFrame(cvcapture: *mut CvCapture) -> c_int;

    pub fn cvRetrieveFrame(cvcapture: *mut CvCapture, streamidx: c_int) -> *mut IplImage;
}


/*
pub struct Video {
    cvcapture: Box<CvCapture>
}

pub fn from_file(fname: &str) -> *mut CvCapture {

    unsafe {

    }
}
*/


#[cfg(test)]
mod tests {
    extern crate libc;

    use self::libc::{c_char, c_int};
    use super::*;

    #[test]
    fn test_cv_capture_from_file() {
        unsafe {
            // Opening a file that does not exist should result in null.
            let c = cvCreateFileCapture("xxxxxxxxxxxxxxx\0".as_ptr() as *const c_char);
            assert!(c.is_null());

            // Open an existing file.
            let d = cvCreateFileCapture("datasets/testing/tree.mp4\0".as_ptr() as *const c_char);
            assert!(!d.is_null());

            // Check that we get all frames of a video of 50 frames.
            for _ in (0..50) {
                let i = cvGrabFrame(d);
                assert!(i != 0);
                let k = cvRetrieveFrame(d, 0 as c_int);
                assert!(!k.is_null());

                assert_eq!((*k).width, 1400);
                assert_eq!((*k).height, 1296);
                assert_eq!((*k).nchannels, 3);
            }

            let i = cvGrabFrame(d);
            assert!(i == 0);
            let k = cvRetrieveFrame(d, 0 as c_int);
            assert!(k.is_null());
        }
    }
}
