//! <font color="red"><b>Experimental</b></font> module for image and video manipulation.
//!
//! Interfaces may change in the future.

// TODO remove memory leaks

extern crate libc;
extern crate rand;

mod bindings;

use std::fmt;
use self::libc::{c_char, c_int, c_float, c_double};
use self::rand::{thread_rng, Rng};

use self::bindings::*;

pub enum FontFace {
    CvFontHersheyComplex,
    CvFontHersheyComplexSmall,
    CvFontHersheyDuplex,
    CvFontHersheyPlain,
    CvFontHersheyScriptComplex,
    CvFontHersheyScriptSimplex,
    CvFontHersheySimplex,
    CvFontHersheyTriplex
}

// ----------------------------------------------------------------------------

/// Font to draw text on images.
pub struct Font {
    font: Box<CvFont>
}

impl Font {

    // TODO parameters
    pub fn new(font: FontFace) -> Font {

        let mut f = Box::new(CvFont {
            namefont: 0 as *const c_char,
            color: CvScalar {
                val: [0 as c_double, 0 as c_double, 0 as c_double, 0 as c_double]
            },
            font_face: 0 as c_int,
            ascii: 0 as *const c_int,
            greek: 0 as *const c_int,
            cyrillic: 0 as *const c_int,
            hscale: 0 as c_float,
            vscale: 0 as c_float,
            shear: 0 as c_float,
            thickness: 0 as c_int,
            dx: 0 as c_float,
            line_type: 0 as c_int
        });

        let font_face: c_int = match font {
            FontFace::CvFontHersheySimplex => 0,
            FontFace::CvFontHersheyComplex => 3,
            FontFace::CvFontHersheyComplexSmall => 5,
            FontFace::CvFontHersheyDuplex => 2,
            FontFace::CvFontHersheyPlain => 1,
            FontFace::CvFontHersheyScriptComplex => 7,
            FontFace::CvFontHersheyScriptSimplex => 6,
            FontFace::CvFontHersheyTriplex => 4,
        };

        unsafe {
            cvInitFont(
                (&mut *f) as *mut CvFont, font_face, 1 as c_double, 1 as c_double,
                0 as c_double, 1 as c_int, 8 as c_int
            );
        }

        Font {
            font: f
        }
    }
}

// ----------------------------------------------------------------------------

/// A window to display an image.
pub struct Window {
    name: String
}

impl Window {
    // TODO destroyWindow

    /// Creates a new window for displaying an image.
    pub fn new() -> Window {
        unsafe {
            let mut s: String = thread_rng().gen_ascii_chars().take(30).collect();
            s.push('\0');
            cvNamedWindow(s.as_ptr() as *const c_char, CV_WINDOW_AUTOSIZE);
            Window {
                name: s
            }
        }
    }

    /// Displays the given image in the window.
    ///
    /// ```ignore
    /// # #[macro_use] extern crate rustml;
    /// use rustml::opencv::*;
    ///
    /// # fn main() {
    /// let img = RgbImage::from_file("datasets/testing/tree.png").unwrap();
    /// Window::new()
    ///     .show_image(&img)
    ///     .wait_key(0);
    /// # }
    /// ```
    pub fn show_image<T: Image>(&self, img: &T) -> &Self {
        unsafe {
            cvShowImage(
                self.name.as_ptr() as *const c_char, 
                img.buffer()       as *const CvArr
            );
        }
        self
    }

    /// Waits for the specified amount of time in milliseconds or until
    /// a key is pressed if `delay` is zero.
    pub fn wait_key(&self, delay: i32) {
        unsafe {
            cvWaitKey(delay as c_int);
            cvDestroyWindow(self.name.as_ptr() as *const c_char);
        }
    }
}

// ------------------------------------------------------------------

/// Represents a pixel with a red, green and blue component.
pub struct Rgb {
    /// Value for red.
    pub r: u8,
    /// Value for green.
    pub g: u8,
    /// Value for blue.
    pub b: u8
}

/// Implementation of `Display`. The format is `(red, green, blue)`.
impl fmt::Display for Rgb {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} {} {})", self.r, self.g, self.b)
    }
}

// ----------------------------------------------------------------------------

/// Represents a pixel with only one component.
pub struct GrayValue {
    pub val: u8,
}

/// Implementation of `Display`. The format is `(value)`.
impl fmt::Display for GrayValue {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({})", self.val)
    }
}

// ----------------------------------------------------------------------------

/// Trait for images.
pub trait Image {
    /// Creates a new image with the given width and height.
    fn new(width: usize, height: usize) -> Self;

    /// Returns the internal representation of the image.
    fn buffer(&self) -> *const IplImage;

    /// Returns the RGB value of the pixel at location `(x, y)`.
    ///
    /// If the image is a grayscale image all color components have the same value.
    fn pixel_as_rgb(&self, x: usize, y: usize) -> Option<Rgb>;

    /// Sets the pixel at the location `(x, y)` to the value specified by `p`.
    ///
    /// If the image is a frayscale image the following grayscale value is
    /// set: val = 0.299 * red + 0.587 * green + 0.114 * blue
    fn set_pixel_from_rgb(&self, x: usize, y: usize, p: &Rgb);

    // implemenations ----------------------------------------------------

    /// Returns the width of the image.
    fn width(&self) -> usize { unsafe { (*self.buffer()).width as usize } }

    /// Returns the height of the image.
    fn height(&self) -> usize { unsafe { (*self.buffer()).height as usize } }

    /// Returns the color depth of the image.
    fn depth(&self) -> usize { unsafe { (*self.buffer()).depth as usize } }

    /// Internal method which returns the length of one row in bytes.
    ///
    /// Due to the fact that rows might by aligned the number of bytes might
    /// be greater than the width of the image.
    fn widthstep(&self) -> usize { unsafe { (*self.buffer()).widthstep as usize } }

    /// Returns the number of color components used for this image.
    fn channels(&self) -> usize { unsafe { (*self.buffer()).nchannels as usize } }

    /// Writes the image into a file.
    ///
    /// The file format that is written depends on the extension of the filename.
    /// Supported formats are JPEG, PNG, PPM, PGM, PBM. Returns `false` if file
    /// could not be written and `true` on success.
    fn to_file(&self, fname: &str) -> bool {

        let mut s = fname.to_string();
        s.push('\0');
        unsafe {
            let r = cvSaveImage(
                s.as_ptr()      as *const c_char, 
                self.buffer()   as *const CvArr,
                0               as *const c_int
            );
            r != 0
        }
    }

    /// Copies an area from the given image at location `(x,y )` with width `width`
    /// and height `height` into this image at position `(dstx, dsty)`.
    fn copy_from<T: Image>(&mut self, 
            img: &T, x: usize, y: usize, width: usize, height: usize, // source
            dstx: usize, dsty: usize) { // dst

        // TODO bounce checking, error handling, performance
        for iy in (0..height) {
            for ix in (0..width) {
                self.set_pixel_from_rgb(
                    dstx + ix, dsty + iy, &img.pixel_as_rgb(x + ix, y + iy).unwrap()
                );
            }
        }
    }

    /// Draws text on the image at position `(x, y)`.
    fn draw_text(&mut self, txt: &str, x: usize, y: usize, font: &Font) {

        let mut s = txt.to_string();
        s.push('\0');

        let p = CvPoint {
            x: x as c_int,
            y: y as c_int
        };

        // TODO color
        let sc = CvScalar {
            val: [255 as c_double, 255 as c_double, 255 as c_double, 255 as c_double]
        };

        unsafe {
            cvPutText(
                self.buffer() as *mut CvArr, 
                s.as_ptr()    as *const c_char,
                p,
                & *(font.font) as *const CvFont,
                sc
            );
        }
    }

    /// Creates a new image by organizing the given list of images into a grid.
    ///
    /// All images must have the same size, otherwise returns `None`. The number
    /// of columns of the grid is specified with the parameter `cols`. The
    /// parameter `space` specifies the number of pixels between each image that
    /// is used to separate them.
    fn grid(images: &Vec<Self>, cols: usize, space: usize) -> Option<Self>;

}

impl Image for GrayImage {
    // TODO refactoring

    fn new(w: usize, h: usize) -> GrayImage {

        let siz = CvSize {
            width: w as c_int,
            height: h as c_int
        };
        GrayImage { 
            iplimage: unsafe { cvCreateImage(siz, 8, 1) } 
        }
    }

    fn grid(images: &Vec<Self>, cols: usize, space: usize) -> Option<GrayImage> {

        grid::<GrayImage>(images, cols, space)
    }

    fn buffer(&self) -> *const IplImage { self.iplimage }

    fn pixel_as_rgb(&self, x: usize, y: usize) -> Option<Rgb> { 
        
        unsafe {
            let img = &(*self.buffer());
            let off = (y * self.widthstep() + x) as isize;

            if off < 0 || off >= (img.imagesize as isize) { 
                return None; 
            }
            
            let gr = *img.imagedata.offset(off) as u8;

            Some(Rgb {
                r: gr,
                g: gr,
                b: gr
            })
        }
    }

    fn set_pixel_from_rgb(&self, x: usize, y: usize, p: &Rgb) {

        let g =
            0.299 * (p.r as f64) +
            0.587 * (p.g as f64) +
            0.114 * (p.b as f64);

        unsafe {
            let img = &(*self.buffer());
            let off = (y * self.widthstep() + x) as isize;

            let p = img.imagedata.offset(off) as *mut u8;
            *p = g as u8;
        }
    }
}

impl Image for RgbImage {
    // TODO refactoring

    fn new(w: usize, h: usize) -> RgbImage {

        let siz = CvSize {
            width: w as c_int,
            height: h as c_int
        };
        RgbImage { 
            iplimage: unsafe { cvCreateImage(siz, 8, 3) } 
        }
    }

    fn grid(images: &Vec<Self>, cols: usize, space: usize) -> Option<RgbImage> {

        grid::<RgbImage>(images, cols, space)
    }

    fn buffer(&self) -> *const IplImage { self.iplimage }

    fn pixel_as_rgb(&self, x: usize, y: usize) -> Option<Rgb> { 

        unsafe {
            let img = &(*self.buffer());
            let off = (y * self.widthstep() + x * 3) as isize;

            if off < 0 || off + 2 >= (img.imagesize as isize) { 
                return None; 
            }
            
            Some(Rgb {
                b: *img.imagedata.offset(off) as u8,
                g: *img.imagedata.offset(off + 1) as u8,
                r: *img.imagedata.offset(off + 2) as u8
            })
        }
    }

    fn set_pixel_from_rgb(&self, x: usize, y: usize, px: &Rgb) {

        unsafe {
            let img = &(*self.buffer());
            let off = (y * self.widthstep() + x * 3) as isize;

            let p = img.imagedata.offset(off) as *mut u8;
            let q = img.imagedata.offset(off + 1) as *mut u8;
            let r = img.imagedata.offset(off + 2) as *mut u8;
            *p = px.b;
            *q = px.g;
            *r = px.r;
        }
    }
}

// ----------------------------------------------------------------------------

/// Creates a new image by organizing the given list of images into a grid.
///
/// All images must have the same size, otherwise returns `None`. The number
/// of columns of the grid is specified with the parameter `cols`. The
/// parameter `space` specifies the number of pixels between each image that
/// is used to separate them.
fn grid<T: Image>(images: &Vec<T>, cols: usize, space: usize) -> Option<T> {

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

    let mut dst = T::new(w, h);
    let mut col = 0;
    let mut row = 0;

    for img in images {
        if img_width != img.width() || img_height != img.height() {
            return None;
        }
        dst.copy_from(
            img, 0, 0, img_width, img_height, col * (img_width + space), row * (img_height + space)
        );
        col += 1;
        if col >= cols {
            col = 0;
            row += 1;
        }
    }
    Some(dst)
}

// ------------------------------------------------------------------

pub struct RgbImage {
    iplimage: *const IplImage
}

impl RgbImage {

    pub fn from_file(fname: &str) -> Option<RgbImage> {

        unsafe {
            let mut s = fname.to_string();
            s.push('\0');
            // 1 = return a 3-channel color imge
            // http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#imread
            let r = cvLoadImage(s.as_ptr() as *const c_char, 1 as c_int);
            let i = RgbImage{ iplimage: r };

            if r.is_null() || i.depth() != 8 || i.channels() != 3 {
                return None;
            }
            Some(i)
        }
    }

    pub fn from_raw(image: *const IplImage) -> RgbImage {

        // TODO convert images
        unsafe {
            assert!((*image).depth == 8 && (*image).nchannels == 3);
            RgbImage {
                iplimage: image
            }
        }
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<Rgb> {

        self.pixel_as_rgb(x, y)
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
        // TODO background
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

    pub fn pixel(&self, x: usize, y: usize) -> Option<GrayValue> {

        match self.pixel_as_rgb(x, y) {
            Some(p) => {
                Some(GrayValue {
                    val: p.r
                })
            }
            _ => None
        }
    }

    // TODO test + bounce checking, duplicated code
    pub fn set_pixel(&mut self, x: usize, y: usize, newval: u8) {

        unsafe {
            let img = &(*self.buffer());
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
                if mask.pixel(x, y).unwrap().val == 255 {
                    self.set_pixel(x, y, newval);
                }
            }
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
                v.push(self.pixel(j, i).unwrap().val);
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

    pub fn pixel_iter(&self) -> GrayValueIterator {

        GrayValueIterator {
            x: 0,
            y: 0,
            image: self
        }
    }
}

pub struct GrayValueIterator<'t> {
    x: usize,
    y: usize,
    image: &'t GrayImage
}

impl <'t> Iterator for GrayValueIterator<'t> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {

        if self.x >= self.image.width() {
            self.x = 0;
            self.y += 1;
        }

        if self.y >= self.image.height() {
            return None;
        }

        self.x += 1;
        Some(self.image.pixel(self.x - 1, self.y).unwrap().val)
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

            if self.mask.pixel(self.x, self.y).unwrap().val != 0 {
                let r = self.src.pixel(self.x, self.y).unwrap().val;
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

    pub fn color_frame_iter(&self) -> VideoColorFrameIterator {
        VideoColorFrameIterator {
            video: self,
        }
    }

    pub fn gray_frame_iter(&self) -> VideoGrayFrameIterator {
        VideoGrayFrameIterator {
            video: self,
        }
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        unsafe { cvReleaseCapture(&self.cvcapture); }
    }
}


pub struct VideoColorFrameIterator<'q> {
    video: &'q Video,
}

impl <'q> Iterator for VideoColorFrameIterator<'q> {
    type Item = RgbImage;

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
                            Some(RgbImage::from_raw(f))
                        }
                    }
                }
            }
        }
    }
}

pub struct VideoGrayFrameIterator<'q> {
    video: &'q Video,
}

impl <'q> Iterator for VideoGrayFrameIterator<'q> {
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
    use super::bindings::*;

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
        assert!(RgbImage::from_file("xxxxxxxxxxxx.png").is_none());
    }

    #[test]
    fn test_colorimage() {
        let i = RgbImage::from_file("datasets/testing/colors.png").unwrap();
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

        let img = RgbImage::from_file("datasets/testing/tree.png").unwrap();
        assert!(img.to_file("/tmp/ab.jpg"));
        // the following test should fail because the directory does not
        // exist
        assert!(!img.to_file("datasets/nulldir/ab.jpg"));
    }

    #[test]
    fn test_font() {

        let mut img = RgbImage::new(300, 300);
        let f = vec![
            Font::new(FontFace::CvFontHersheySimplex),
            Font::new(FontFace::CvFontHersheyComplex),
            Font::new(FontFace::CvFontHersheyComplexSmall),
            Font::new(FontFace::CvFontHersheyDuplex),
            Font::new(FontFace::CvFontHersheyPlain),
            Font::new(FontFace::CvFontHersheyScriptComplex),
            Font::new(FontFace::CvFontHersheyScriptSimplex),
            Font::new(FontFace::CvFontHersheyTriplex),
        ];
        for i in (0..8) {
            img.draw_text("hallo", 10, i * 20 + 20, f.get(i).unwrap());
        }
        img.to_file("/tmp/blabla.jpg");
    }
}
