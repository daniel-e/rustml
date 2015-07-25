//! Bindings for OpenCV.

extern crate libc;

use std::fmt;
use self::libc::{c_char, c_int, c_void};

#[repr(C)]
pub struct CvCapture;

#[repr(C)]
pub struct CvArr;

#[repr(C)]
pub struct CvSize {
    width: c_int,
    height: c_int
}

// http://docs.opencv.org/modules/core/doc/old_basic_structures.html
#[repr(C)]
pub struct IplImage {
    nsize: c_int,
    id: c_int,
    pub nchannels: c_int,
    alphachannel: c_int,
    pub depth: c_int,  // iki.icub.org/yarpdoc/IplImage_8h.html
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
    pub imagesize: c_int,
    pub imagedata: *mut c_char,
    pub widthstep: c_int,
    bordermode: [c_int; 4],
    borderconst: [c_int; 4],
    imagedataorigin: *mut c_char
}

const CV_BGR2GRAY: c_int = 6;

#[link(name = "opencv_highgui")]
extern {
    pub fn cvCreateFileCapture(fname: *const c_char) -> *const CvCapture;

    pub fn cvGrabFrame(cvcapture: *const CvCapture) -> c_int;

    pub fn cvRetrieveFrame(cvcapture: *const CvCapture, streamidx: c_int) -> *const IplImage;

    pub fn cvReleaseCapture(cvcapture: *const *const CvCapture);

    pub fn cvLoadImage(fname: *const c_char, iscolor: c_int) -> *const IplImage;

    pub fn cvSaveImage(fname: *const c_char, img: *const CvArr, params: *const c_int) -> c_int;
}

#[link(name = "opencv_core")]
extern {
    pub fn cvCreateImage(siz: CvSize, depth: c_int, channels: c_int) -> *const IplImage;
}

#[link(name = "opencv_imgproc")]
extern {
    pub fn cvCvtColor(src: *const CvArr, dst: *mut CvArr, code: c_int);
}

// ----------------------------------------------------------------------------

pub fn grid(images: &Vec<GrayImage>, cols: usize, space: usize) -> Option<GrayImage> {

    if images.len() == 0 || cols == 0 {
        return None;
    }

    let mut rows = images.len() / cols;
    if rows * cols < images.len() {
        rows += 1;
    }

    let img_width = images.last().unwrap().width();
    let img_height = images.last().unwrap().height();
    let w = img_width * cols + (cols - 1) * space;
    let h = img_height * rows + (rows - 1) * space;

    let siz = CvSize {
        width: w as c_int,
        height: h as c_int
    };
    let mut dst = GrayImage { iplimage: unsafe { cvCreateImage(siz, 8, 1) } };

    let mut col = 0;
    let mut row = 0;

    for img in images {
        if img_width != img.width() || img_height != img.height() {
            return None;
        }
        for y in (0..img_height) {
            for x in (0..img_width) {
                dst.set_pixel(
                    x + col * (img_width + space), y + row * (img_height + space), 
                    img.pixel(x, y).unwrap().g
                );
            }
        }
        col += 1;
        if col >= cols {
            col = 0;
            row += 1;
        }
    }

    Some(dst)
}

// ----------------------------------------------------------------------------

pub struct GrayPixel {
    pub g: u8,
}

impl fmt::Display for GrayPixel {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({})", self.g)
    }
}

// ------------------------------------------------------------------

pub struct ColorPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8
}

impl fmt::Display for ColorPixel {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} {} {})", self.r, self.g, self.b)
    }
}

// ------------------------------------------------------------------

pub struct ColorImage {
    iplimage: *const IplImage
}

impl ColorImage {

    pub fn from_file(fname: &str) -> Option<ColorImage> {

        unsafe {
            let mut s = fname.to_string();
            s.push('\0');
            // 1 = return a 3-channel color imge
            // http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#imread
            let r = cvLoadImage(s.as_ptr() as *const c_char, 1 as c_int);
            let i = ColorImage{ iplimage: r };

            if r.is_null() || i.depth() != 8 || i.channels() != 3 {
                return None;
            }
            Some(i)
        }
    }

    pub fn from_raw(image: *const IplImage) -> ColorImage {

        // TODO convert images
        unsafe {
            assert!((*image).depth == 8 && (*image).nchannels == 3);
            ColorImage {
                iplimage: image
            }
        }
    }

    pub fn to_file(&self, fname: &str) -> bool {

        let mut s = fname.to_string();
        s.push('\0');
        unsafe {
            let r = cvSaveImage(
                s.as_ptr()      as *const c_char, 
                self.iplimage   as *const CvArr,
                0               as *const c_int
            );
            r != 0
        }
    }

    pub fn width(&self) -> usize { unsafe { (*self.iplimage).width as usize } }
    pub fn height(&self) -> usize { unsafe { (*self.iplimage).height as usize } }
    pub fn depth(&self) -> usize { unsafe { (*self.iplimage).depth as usize } }
    pub fn widthstep(&self) -> usize { unsafe { (*self.iplimage).widthstep as usize } }
    pub fn channels(&self) -> usize { unsafe { (*self.iplimage).nchannels as usize } }

    pub fn pixel(&self, x: usize, y: usize) -> Option<ColorPixel> {

        unsafe {
            let img = &(*self.iplimage);
            let off = (y * self.widthstep() + x * 3) as isize;

            if off < 0 || off + 2 >= (img.imagesize as isize) { 
                return None; 
            }
            
            Some(ColorPixel {
                b: *img.imagedata.offset(off) as u8,
                g: *img.imagedata.offset(off + 1) as u8,
                r: *img.imagedata.offset(off + 2) as u8
            })
        }
    }

}

// ------------------------------------------------------------------

pub struct GrayImage {
    iplimage: *const IplImage
}

impl GrayImage {

    pub fn from_file(fname: &str) -> Option<GrayImage> {

        unsafe {
            let mut s = fname.to_string();
            s.push('\0');
            // 0 = return a grayscale image
            // http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#imread
            let r = cvLoadImage(s.as_ptr() as *const c_char, 0 as c_int);

            if r.is_null() {
                return None;
            }
            Some(GrayImage::from_raw(r))
        }
    }

    pub fn from_slice(v: &[u8], rows: usize, cols: usize) -> Option<GrayImage> {

        if rows * cols != v.len() {
            return None;
        }

        let siz = CvSize {
            width: cols as c_int,
            height: rows as c_int
        };
        let mut dst = GrayImage { iplimage: unsafe { cvCreateImage(siz, 8, 1) } };
        
        for y in (0..rows) {
            for x in (0..cols) {
                dst.set_pixel(x, y, *v.get(y * cols + x).unwrap());
            }
        }
        Some(dst)
    }

    pub fn from_raw(image: *const IplImage) -> GrayImage {

        unsafe {
            if (*image).depth != 8 || (*image).nchannels != 1 {
                let siz = CvSize {
                    width: (*image).width,
                    height: (*image).height
                };
                let ipl = cvCreateImage(siz, 8, 1);
                cvCvtColor(image as *const CvArr, ipl as *mut CvArr, CV_BGR2GRAY);
                GrayImage {
                    iplimage: ipl
                }
            } else {
                GrayImage {
                    iplimage: image
                }
            }
        }
    }

    pub fn width(&self) -> usize { unsafe { (*self.iplimage).width as usize } }
    pub fn height(&self) -> usize { unsafe { (*self.iplimage).height as usize } }
    pub fn depth(&self) -> usize { unsafe { (*self.iplimage).depth as usize } }
    pub fn widthstep(&self) -> usize { unsafe { (*self.iplimage).widthstep as usize } }
    pub fn channels(&self) -> usize { unsafe { (*self.iplimage).nchannels as usize } }

    pub fn pixel(&self, x: usize, y: usize) -> Option<GrayPixel> {

        unsafe {
            let img = &(*self.iplimage);
            let off = (y * self.widthstep() + x) as isize;

            if off < 0 || off >= (img.imagesize as isize) { 
                return None; 
            }
            
            Some(GrayPixel {
                g: *img.imagedata.offset(off) as u8,
            })
        }
    }

    // TODO test + bounce checking
    pub fn set_pixel(&mut self, x: usize, y: usize, newval: u8) {

        unsafe {
            let img = &(*self.iplimage);
            let off = (y * self.widthstep() + x) as isize;

            let p = img.imagedata.offset(off) as *mut u8;
            *p = newval;
        }
    }


    // TODO test
    pub fn set_pixel_mask(&mut self, mask: &GrayImage, newval: u8) {

        // TODO check size of mask
        for y in (0..self.height()) {
            for x in (0..self.width()) {
                if mask.pixel(x, y).unwrap().g == 255 {
                    self.set_pixel(x, y, newval);
                }
            }
        }
    }

    pub fn to_file(&self, fname: &str) -> bool {

        let mut s = fname.to_string();
        s.push('\0');
        unsafe {
            let r = cvSaveImage(
                s.as_ptr()      as *const c_char, 
                self.iplimage   as *const CvArr,
                0               as *const c_int
            );
            r != 0
        }
    }

    pub fn pixels_from_mask_as_u8(&self, i: &GrayImage) -> Option<Vec<u8>> {

        if self.width() != i.width() || self.height() != i.height() {
            return None;
        }

        let mut pixels: Vec<u8> = Vec::new();
        unsafe {
            for y in (0..self.height()) {
                let s: *const c_char = (*self.iplimage).imagedata.offset(y as isize * self.widthstep() as isize);
                let m: *const c_char = (*i.iplimage).imagedata.offset(y as isize * i.widthstep() as isize);
                for x in (0..self.width()) {
                    if *m.offset(x as isize) != 0 {
                        pixels.push(*s.offset(x as isize) as u8);
                    }
                }
            }
        }

        Some(pixels)
    }

    // TODO test
    pub fn rectangle(&self, x: usize, y: usize, width: usize, height: usize) -> Vec<u8> {

        // TODO bounce checking
        let mut v = Vec::new();
        for i in (y..y+height) {
            for j in (x..x+width) {
                v.push(self.pixel(j, i).unwrap().g);
            }
        }
        v
    }

    pub fn mask_iter<'q>(&'q self, i: &'q GrayImage) -> MaskIter {

        MaskIter {
            src: self,
            mask: i,
            x: 0,
            y: 0,
        }
    }
}

pub struct MaskIter<'t> {
    src: &'t GrayImage,
    mask: &'t GrayImage,
    x: usize,
    y: usize,
}

impl <'t> Iterator for MaskIter<'t> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
       
        if self.src.width() != self.mask.width() || self.src.height() != self.mask.height() {
            return None;
        }

        loop {

            if self.x >= self.src.width() {
                self.x = 0;
                self.y += 1;
            }

            if self.y >= self.src.height() {
                return None;
            }

            if self.mask.pixel(self.x, self.y).unwrap().g != 0 {
                let r = self.src.pixel(self.x, self.y).unwrap().g;
                self.x += 1;
                return Some(r);
            }
            self.x += 1;
        }
    }
}


// ------------------------------------------------------------------

pub struct Video {
    pub cvcapture: *const CvCapture
}

impl Video {

    // TODO use path
    pub fn from_file(fname: &str) -> Option<Video> {
        let mut f = fname.to_string();
        f.push('\0');
        unsafe {
            let c = cvCreateFileCapture(f.as_ptr() as *const c_char);
            match c.is_null() {
                true  => None,
                false => {
                    Some(Video {
                        cvcapture: c
                    })
                }
            }
        }
    }

    pub fn color_frame_iter(&self) -> ColorImageIterator {
        ColorImageIterator {
            video: self,
        }
    }

    pub fn gray_frame_iter(&self) -> GrayImageIterator {
        GrayImageIterator {
            video: self,
        }
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        unsafe { cvReleaseCapture(&self.cvcapture); }
    }
}


pub struct ColorImageIterator<'q> {
    video: &'q Video,
}

impl <'q> Iterator for ColorImageIterator<'q> {
    type Item = ColorImage;

    // TODO return a reference; currently it is dangerous because the pointer
    // can become invalid
    fn next(&mut self) -> Option<Self::Item> {

        unsafe {
            let i = cvGrabFrame(self.video.cvcapture);
            match i {
                0 => None,
                _ => {
                    // http://www.cs.indiana.edu/cgi-pub/oleykin/website/OpenCVHelp/ref/OpenCVRef_HighGUI.htm
                    // "the retrieved frame should not be released by the user"
                    let f = cvRetrieveFrame(self.video.cvcapture, 0 as c_int);
                    match f.is_null() {
                        true  => None,
                        false => {
                            Some(ColorImage::from_raw(f))
                        }
                    }
                }
            }
        }
    }
}

pub struct GrayImageIterator<'q> {
    video: &'q Video,
}

impl <'q> Iterator for GrayImageIterator<'q> {
    type Item = GrayImage;

    // TODO return a reference; currently it is dangerous because the pointer
    // can become invalid
    fn next(&mut self) -> Option<Self::Item> {

        unsafe {
            let i = cvGrabFrame(self.video.cvcapture);
            match i {
                0 => None,
                _ => {
                    // http://www.cs.indiana.edu/cgi-pub/oleykin/website/OpenCVHelp/ref/OpenCVRef_HighGUI.htm
                    // "the retrieved frame should not be released by the user"
                    let f = cvRetrieveFrame(self.video.cvcapture, 0 as c_int);
                    match f.is_null() {
                        true  => None,
                        false => {
                            Some(GrayImage::from_raw(f))
                        }
                    }
                }
            }
        }
    }
}



#[cfg(test)]
mod tests {
    extern crate libc;

    use self::libc::{c_char, c_int};
    use super::*;

    #[test]
    fn test_cv_capture_from_file_lowlevel() {
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

            //cvReleaseCapture(&d);
        }
    }

    #[test]
    fn test_cv_capture_pixel_lowlevel() {
        let d = Video::from_file("datasets/testing/colors.mp4").unwrap();

        let i = d.color_frame_iter().next().unwrap();
        assert_eq!(i.channels(), 3);
        assert_eq!(i.widthstep(), 300);
        assert_eq!(i.depth(), 8);

        // check that the red box exists
        for y in (0..45) {
            for x in (0..45) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r > 230);
                assert!(p.g < 20);
                assert!(p.b < 20);
            }
        }
        // check that the white box exists
        for y in (0..45) {
            for x in (55..95) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r > 230);
                assert!(p.g > 230);
                assert!(p.b > 230);
            }
        }
        // check that the black box exists
        for y in (55..100) {
            for x in (0..45) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r < 20);
                assert!(p.g < 20);
                assert!(p.b < 20);
            }
        }
        // check that the green box exists
        for y in (55..100) {
            for x in (55..95) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r < 20);
                assert!(p.g > 230);
                assert!(p.b < 20);
            }
        }
    }

    #[test]
    fn test_video() {
        assert!(Video::from_file("xxxxxxxxxxx.mp4").is_none());

        {
        let c = Video::from_file("datasets/testing/colors.mp4").unwrap();
        let i = c.color_frame_iter();
        assert_eq!(i.count(), 25);
        }

        let c = Video::from_file("datasets/testing/colors.mp4").unwrap();
        let mut k = 0;
        for img in c.color_frame_iter() {
            k += 1;
            let mut p = img.pixel(25, 25).unwrap();
            assert!(p.r > 200 && p.b < 20 && p.g < 20);
            p = img.pixel(75, 25).unwrap();
            assert!(p.r > 200 && p.b > 200 && p.g > 200);
            p = img.pixel(25, 75).unwrap();
            assert!(p.r < 20 && p.b < 20 && p.g < 20);
            p = img.pixel(75, 75).unwrap();
            assert!(p.r < 20 && p.b < 20 && p.g > 200);
        }
        assert_eq!(k, 25);
    }

    #[test]
    fn test_colorimage_error() {
        assert!(ColorImage::from_file("xxxxxxxxxxxx.png").is_none());
    }

    #[test]
    fn test_colorimage() {
        let i = ColorImage::from_file("datasets/testing/colors.png").unwrap();
        assert_eq!(i.width(), 100);
        assert_eq!(i.height(), 100);

        // check that the red box exists
        for y in (0..50) {
            for x in (0..50) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r == 255 && p.b == 0 && p.g == 0);
            }
        }
        for y in (0..50) {
            for x in (50..100) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r == 255 && p.b == 255 && p.g == 255);
            }
        }
        for y in (50..100) {
            for x in (0..50) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r == 0 && p.b == 0 && p.g == 0);
            }
        }
        for y in (50..100) {
            for x in (50..100) {
                let p = i.pixel(x, y).unwrap();
                assert!(p.r == 0 && p.b == 0 && p.g == 255);
            }
        }
    }

    #[test]
    fn test_mask_image() {

        let mask = GrayImage::from_file("datasets/testing/10x10colors_mask.png").unwrap();
        let gray = GrayImage::from_file("datasets/testing/10x10gray.png").unwrap();

        let v = gray.pixels_from_mask_as_u8(&mask);
        assert_eq!(v.unwrap(),
            vec![0x36, 0x36, 0xed, 0x12, 0x12, 0x36, 0x36, 0xff, 0x36, 0x49, 0x00, 0xff]
        );

        let x: Vec<u8> = gray.mask_iter(&mask).collect();
        assert_eq!(x,
            vec![0x36, 0x36, 0xed, 0x12, 0x12, 0x36, 0x36, 0xff, 0x36, 0x49, 0x00, 0xff]
        );
    }

    #[test]
    fn test_mask_video() {

        let video = Video::from_file("datasets/testing/colors.mp4").unwrap();
        let mask = GrayImage::from_file("datasets/testing/colors_mask.png").unwrap();

        let img = video.gray_frame_iter().next().unwrap();
        let pixels = img.mask_iter(&mask).collect::<Vec<u8>>();
        assert_eq!(pixels, vec![
            76, 76, 76, 76,
            255, 255, 255, 255, 255, 255, 255, 255, 255, 
            0, 0, 149, 149, 149, 0, 149, 149, 149, 0, 149, 149, 149, 0
        ]);
    }

    #[test]
    fn test_image_to_file() {

        let img = ColorImage::from_file("datasets/testing/tree.png").unwrap();
        assert!(img.to_file("/tmp/ab.jpg"));
        // the following test should fail because the directory does not
        // exist
        assert!(!img.to_file("datasets/nulldir/ab.jpg"));
    }
}
