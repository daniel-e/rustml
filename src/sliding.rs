//! Sliding windows over strings, bytes and ranges for arbitrary dimensions.
//!
//! A sliding window is often used in image processing, e.g. for object
//! recognition. It is a technique where a small window is moved over the
//! image and some image processing is performed on that sub-image, e.g.
//! to detect objects. A sliding window is also useful for strings to
//! extract all substrings of a fixed size.
//! 
//! A sliding window with a width `w` on an area with width `a` and a
//! delta `d` returns the following set of numbers (i.e. positions of
//! the window): 
//! `{0, d, 2d, ..., w - d + 1}`
//!
//! A sliding window on an area with two dimensions returns a set of
//! tuples of window positions. A sliding window on an area with more
//! then two dimensions returns a set of vectors of window positions.
//!
//! # Examples
//!
//! The recommended way to build a sliding window over an arbitrary number of
//! ranges is to first call the function `builder` and to sequently call the
//! method `add` of the returned instance for each dimension.
//!
//! ```
//! # #[macro_use] extern crate rustml;
//! // Example of a sliding window over two dimensions.
//! use rustml::sliding::*;
//!
//! # fn main() {
//! // Create a sliding window over two dimensions. For the first dimension the
//! // size of the window is 10, the position is incremented by 25 and the
//! // area over which the window is moved is 100. for the second dimension
//! // the size of the window is 10, the position is incremented by 2 and the
//! // area over which the window is moved is 10.
//! let windows = builder().add(100, 10, 25).add(10, 1, 2).to_2d().unwrap();
//! assert_eq!(windows, vec![
//!     (0, 0), (25, 0), (50, 0), (75, 0),
//!     (0, 2), (25, 2), (50, 2), (75, 2),
//!     (0, 4), (25, 4), (50, 4), (75, 4),
//!     (0, 6), (25, 6), (50, 6), (75, 6),
//!     (0, 8), (25, 8), (50, 8), (75, 8)
//! ]);
//! # }
//! ```
//!
//! Another way to build a sliding window over an arbitrary number of
//! ranges is to use the function `sliding_window`.
//!
//! ```
//! # #[macro_use] extern crate rustml;
//! // Example of a sliding window over two dimensions.
//! use rustml::sliding::*;
//!
//! # fn main() {
//! let windows = builder().add(100, 10, 25).add(10, 1, 2).to_vec();
//!
//! let params = vec![
//!     param(100, 10, 25),
//!     param(10, 1, 2)
//! ];
//!
//! assert_eq!(
//!     sliding_window(&params),
//!     windows
//! );
//! # }
//! ```
//!
//!
//! ```
//! # #[macro_use] extern crate rustml;
//! // Example of a sliding window over one dimensions.
//! use rustml::sliding::*;
//!
//! # fn main() {
//! assert_eq!(
//!     builder().add(50, 20, 5).to_1d().unwrap(),
//!     vec![0, 5, 10, 15, 20, 25, 30]
//! );
//! # }
//! ```
//!
//! Sliding window over a string.
//!
//! ```
//! # #[macro_use] extern crate rustml;
//! use rustml::sliding::*;
//!
//! # fn main() {
//! let s = "hello, world!";
//! assert_eq!(
//!     // create a sliding window of size 5
//!     string_slider(s, 5).unwrap().collect::<Vec<&str>>(),
//!     vec![
//!         "hello", "ello,", "llo, ", "lo, w", 
//!         "o, wo", ", wor", " worl", "world", "orld!"
//!     ]
//! );
//! # }
//! ```

extern crate num;
use self::num::iter;

/// Contains the parameters for one dimensions of a sliding window.
///
/// Usually the function [param](fn.param.html) is used to create a
/// set of parameters but a set of parameters can be created with the
/// method `new` as well.
///
/// A sliding window with a width `w` on an area with width `a` and a
/// delta `d` returns the following set of numbers: <br>
/// `{0, d, 2d, ..., w - d + 1}`
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::sliding::*;
/// # fn main() {
/// assert_eq!(
///     DimensionParameters::new(100, 10, 5),
///     param(100, 10, 5)
/// );
/// # }
/// ```
///
#[derive(Copy, Clone, Debug)]
pub struct DimensionParameters {
    /// The size of the area on which the sliding window is moved.
    pub area_width: usize,
    /// The size of the sliding window.
    pub window_width: usize,
    /// The increment that is used to move the sliding window.
    pub delta: usize
}

impl PartialEq<DimensionParameters> for DimensionParameters {

    fn eq(&self, other: &DimensionParameters) -> bool {
        self.area_width == other.area_width &&
        self.window_width == other.window_width &&
        self.delta == other.delta
    }
}

impl DimensionParameters {
    pub fn new(area_width: usize, window_width: usize, delta: usize) -> DimensionParameters {
        DimensionParameters {
            area_width: area_width,
            window_width: window_width,
            delta: delta
        }
    }
}

/// Creates a set of parameters for one dimension of a sliding window.
///
/// This is a convenient function which can be used instead of the `new`
/// method of the [DimensionParameters](struct.DimensionParameters.html) structure.
/// # Example
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::sliding::*;
/// # fn main() {
/// assert_eq!(
///     DimensionParameters::new(100, 10, 5),
///     param(100, 10, 5)
/// );
/// # }
/// ```
pub fn param(area_width: usize, window_width: usize, delta: usize) -> DimensionParameters {
    DimensionParameters {
        area_width: area_width,
        window_width: window_width,
        delta: delta
    }
}

/// A builder to create a sliding window over an arbitrary number of dimensions.
pub struct SlidingWindowBuilder {
    dimensions: Vec<DimensionParameters>
}

impl SlidingWindowBuilder {

    /// Adds an additional dimension to the sliding window and returns
    /// the new `SlidingWindowBuilder`.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::sliding::*;
    ///
    /// # fn main() {
    /// assert_eq!(
    ///     builder().add(10, 3, 2).to_1d().unwrap(),
    ///     vec![0, 2, 4, 6]
    /// );
    /// # }
    /// ```
    pub fn add(&self, area_width: usize, window_width: usize, delta: usize) -> SlidingWindowBuilder {

        let mut v = self.dimensions.clone();
        v.push(DimensionParameters::new(area_width, window_width, delta));
        SlidingWindowBuilder {
            dimensions: v
        }
    }

    /// Returns a vector of the positions of a sliding window with arbitrary dimensions.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::sliding::*;
    ///
    /// # fn main() {
    /// assert_eq!(
    ///     builder().add(10, 3, 2).add(10, 1, 4).to_vec(),
    ///     vec![
    ///         vec![0, 0], vec![2, 0], vec![4, 0], vec![6, 0],
    ///         vec![0, 4], vec![2, 4], vec![4, 4], vec![6, 4],
    ///         vec![0, 8], vec![2, 8], vec![4, 8], vec![6, 8]
    ///     ]
    /// );
    /// # }
    /// ```
    pub fn to_vec(&self) -> Vec<Vec<usize>> {
        sliding_window(&self.dimensions)
    }

    /// Returns a vector of the positions of a one dimensional sliding window.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::sliding::*;
    ///
    /// # fn main() {
    /// assert_eq!(
    ///     builder().add(10, 3, 2).to_1d().unwrap(),
    ///     vec![0, 2, 4, 6]
    /// );
    /// # }
    /// ```
    pub fn to_1d(&self) -> Option<Vec<usize>> {

        if self.dimensions.len() != 1 {
            None
        } else {
            Some(sliding_window_1d(&self.dimensions[0]))
        }
    }

    /// Returns a vector of tuples of the positions of a two dimensional
    /// sliding windoe.
    ///
    /// ```
    /// # #[macro_use] extern crate rustml;
    /// use rustml::sliding::*;
    ///
    /// # fn main() {
    /// assert_eq!(
    ///     builder().add(10, 3, 2).add(10, 1, 4).to_2d().unwrap(),
    ///     vec![
    ///         (0, 0), (2, 0), (4, 0), (6, 0),
    ///         (0, 4), (2, 4), (4, 4), (6, 4),
    ///         (0, 8), (2, 8), (4, 8), (6, 8)
    ///     ]
    /// );
    /// # }
    /// ```
    pub fn to_2d(&self) -> Option<Vec<(usize, usize)>> {

        if self.dimensions.len() != 2 {
            None
        } else {
            Some(sliding_window_2d(&self.dimensions[0], &self.dimensions[1]))
        }
    }
}

/// A function to comfortably create a sliding window over an arbitrary number 
/// of dimensions.
///
/// The returned `StringWindowBuilder` can then be used to add dimensions via
/// its `add` method.
pub fn builder() -> SlidingWindowBuilder {
    SlidingWindowBuilder {
        dimensions: vec![]
    }
}

/// Function to create a sliding window over an arbitrary number of dimensions.
pub fn sliding_window(dp: &[DimensionParameters]) -> Vec<Vec<usize>> {

    let mut r: Vec<Vec<usize>> = Vec::new();

    if dp.len() == 1 {
        for i in iter::range_step_inclusive(0, dp[0].area_width - dp[0].window_width, dp[0].delta) {
            r.push(vec![i]);
        }
    } else {
        let (x, y) = dp.split_at(1);
        let v = sliding_window(y);
        for ref values in v {
            for i in iter::range_step_inclusive(0, x[0].area_width - x[0].window_width, x[0].delta) {
                let mut k: Vec<usize> = Vec::new();
                k.push(i);
                for item in values {
                    k.push(*item);
                }
                r.push(k);
            }
        }
    }
    r
}

/// Function to create a sliding window over one dimension.
pub fn sliding_window_1d(dp: &DimensionParameters) -> Vec<usize> {

    sliding_window(&[*dp]).iter().flat_map(|x| x.iter()).cloned().collect::<Vec<usize>>()
}

/// Funciton to create a sliding window over two dimensions.
pub fn sliding_window_2d(x: &DimensionParameters, y: &DimensionParameters) -> Vec<(usize, usize)> {

    sliding_window(&[*x, *y])
        .iter().map(|v| (v[0], v[1])).collect::<Vec<(usize, usize)>>()
}

// -------------------------------------------------------------------------

/// Sliding window iterator of fixed size over a string.
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::sliding::*;
///
/// # fn main() {
/// let s = "hello";
/// assert_eq!(
///     string_slider(&s, 3).unwrap().collect::<Vec<&str>>(),
///     vec![
///         "hel",
///         "ell",
///         "llo"
///     ]
/// );
/// # }
/// ```
pub struct StringSlider<'a> {
    s: &'a str,
    winlen: usize,
    pos: usize
}

impl <'a> Iterator for StringSlider<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.winlen - 1 >= self.s.len() {
            return None;
        }

        self.pos += 1;
        // not character but byte
        // https://users.rust-lang.org/t/how-to-get-a-substring-of-a-string/1351/7
        Some(&self.s[(self.pos - 1)..(self.pos - 1 + self.winlen)])
    }
}

/// Creates a sliding window of fixed size over a string.
///
/// Returns `None` if `winlen` is zero, otherwise a `StringSlider` is
/// returned that implements `Iterator` to iterate through the
/// string.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::sliding::*;
///
/// # fn main() {
/// let s = "hello";
/// assert_eq!(
///     string_slider(&s, 3).unwrap().collect::<Vec<&str>>(),
///     vec![
///         "hel",
///         "ell",
///         "llo"
///     ]
/// );
/// # }
/// ```
pub fn string_slider<'a>(s: &'a str, winlen: usize) -> Option<StringSlider<'a>> {

    if winlen == 0 {
        return None;
    }

    Some(StringSlider {
        s: s,
        winlen: winlen,
        pos: 0
    })
}

// -------------------------------------------------------------------------

/// Sliding window iterator of fixed size over bytes created with the function
/// [byte_slider](fn.byte_slider.html).
///
/// # Example
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::sliding::*;
///
/// # fn main() {
/// let v = vec![1, 2, 3, 4, 5, 6];
/// assert_eq!(
///     byte_slider(&v, 3).unwrap().collect::<Vec<&[u8]>>(),
///     vec![
///         &[1, 2, 3],
///         &[2, 3, 4],
///         &[3, 4, 5],
///         &[4, 5, 6]
///     ]
/// );
/// # }
/// ```
pub struct ByteSlider<'a> {
    s: &'a [u8],
    winlen: usize,
    pos: usize
}

impl <'a> Iterator for ByteSlider<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.winlen - 1 >= self.s.len() {
            return None;
        }

        self.pos += 1;
        // not character but byte
        // https://users.rust-lang.org/t/how-to-get-a-substring-of-a-string/1351/7
        Some(&self.s[(self.pos - 1)..(self.pos - 1 + self.winlen)])
    }
}

/// Creates a sliding window iterator of fixed size over bytes.
///
/// Returns `None` if `winlen` is zero, otherwise a `ByteSlider` is
/// returned that implements `Iterator` to iterate through the
/// bytes.
///
/// ```
/// # #[macro_use] extern crate rustml;
/// use rustml::sliding::*;
///
/// # fn main() {
/// let v = vec![1, 2, 3, 4, 5, 6];
/// assert_eq!(
///     byte_slider(&v, 3).unwrap().collect::<Vec<&[u8]>>(),
///     vec![
///         &[1, 2, 3],
///         &[2, 3, 4],
///         &[3, 4, 5],
///         &[4, 5, 6]
///     ]
/// );
/// # }
/// ```
pub fn byte_slider<'a>(s: &'a [u8], winlen: usize) -> Option<ByteSlider<'a>> {

    match winlen {
        0 => None,
        _ => Some(ByteSlider {
            s: s,
            winlen: winlen,
            pos: 0
        })
    }
}

// -------------------------------------------------------------------------
// TODO tests
