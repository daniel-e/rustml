extern crate libc;
use self::libc::{c_char, c_int, c_void, c_float, c_double};

pub enum CvCapture {}

pub enum CvArr {}

#[repr(C)]
pub struct CvSize {
    pub width: c_int,
    pub height: c_int
}

#[repr(C)]
pub struct CvScalar {
    pub val: [c_double; 4]
}

#[repr(C)]
pub struct CvPoint {
    pub x: c_int,
    pub y: c_int
}

#[repr(C)]
pub struct CvFont {
    pub namefont: *const c_char,
    pub color: CvScalar,
    pub font_face: c_int,
    pub ascii: *const c_int,
    pub greek: *const c_int,
    pub cyrillic: *const c_int,
    pub hscale: c_float,
    pub vscale: c_float,
    pub shear: c_float,
    pub thickness: c_int,
    pub dx: c_float,
    pub line_type: c_int
}

// http://docs.opencv.org/modules/core/doc/old_basic_structures.html
#[repr(C)]
pub struct IplImage {
    pub nsize: c_int,
    pub id: c_int,
    pub nchannels: c_int,
    pub alphachannel: c_int,
    pub depth: c_int,  // iki.icub.org/yarpdoc/IplImage_8h.html
    pub colormodel: [c_char; 4],
    pub channelseq: [c_char; 4],
    pub dataorder: c_int,
    pub origin: c_int,
    pub align: c_int,
    pub width: c_int,
    pub height: c_int,
    pub roi: *mut c_void, // actually it is not a void pointer
    pub maskroi: *mut c_void, // actually it is not a void pointer
    pub imageid: *mut c_void,
    pub titleinfo: *mut c_void,
    pub imagesize: c_int,
    pub imagedata: *mut c_char,
    pub widthstep: c_int,
    pub bordermode: [c_int; 4],
    pub borderconst: [c_int; 4],
    pub imagedataorigin: *mut c_char
}

pub const CV_BGR2GRAY: c_int = 6;
pub const CV_WINDOW_AUTOSIZE: c_int = 1;

#[link(name = "opencv_highgui")]
extern {
    pub fn cvCreateFileCapture(fname: *const c_char) -> *const CvCapture;

    pub fn cvGrabFrame(cvcapture: *const CvCapture) -> c_int;

    pub fn cvRetrieveFrame(cvcapture: *const CvCapture, streamidx: c_int) -> *const IplImage;

    pub fn cvReleaseCapture(cvcapture: *const *const CvCapture);

    pub fn cvLoadImage(fname: *const c_char, iscolor: c_int) -> *const IplImage;

    pub fn cvSaveImage(fname: *const c_char, img: *const CvArr, params: *const c_int) -> c_int;

    pub fn cvNamedWindow(name: *const c_char, flags: c_int);

    pub fn cvShowImage(winname: *const c_char, img: *const CvArr);

    pub fn cvWaitKey(delay: c_int) -> c_int;

    pub fn cvDestroyWindow(winname: *const c_char);

    pub fn cvInitFont(font: *mut CvFont, font_face: c_int, hscale: c_double, vscale: c_double,
                      shear: c_double, thickness: c_int, line_type: c_int);
}

#[link(name = "opencv_core")]
extern {
    pub fn cvCreateImage(siz: CvSize, depth: c_int, channels: c_int) -> *const IplImage;

    pub fn cvPutText(img: *mut CvArr, text: *const c_char, org: CvPoint, 
                     font: *const CvFont, color: CvScalar);
}

#[link(name = "opencv_imgproc")]
extern {
    pub fn cvCvtColor(src: *const CvArr, dst: *mut CvArr, code: c_int);

    pub fn cvRectangle(img: *mut CvArr, p1: CvPoint, p2: CvPoint, color: CvScalar,
                       thickness: c_int, line_type: c_int, shift: c_int);
}


