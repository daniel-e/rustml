//! Sliding windows over strings, bytes and over ranges for arbitrary dimensions.
//!
//! A sliding window is often used in image processing.
//!
//! # Examples
//!
//! The recommended way to build a sliding window over ranges.
//!
//! ```
//! # #[macro_use] extern crate rustml;
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

extern crate num;
use self::num::iter;

/// Contains the parameters for one dimensions of a sliding window.
#[derive(Copy, Clone)]
pub struct DimensionParameters {
    pub area_width: usize,
    pub window_width: usize,
    pub delta: usize
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

    pub fn add(&self, area_width: usize, window_width: usize, delta: usize) -> SlidingWindowBuilder {

        let mut v = self.dimensions.clone();
        v.push(DimensionParameters::new(area_width, window_width, delta));
        SlidingWindowBuilder {
            dimensions: v
        }
    }

    pub fn to_vec(&self) -> Vec<Vec<usize>> {
        sliding_window(&self.dimensions)
    }

    pub fn to_1d(&self) -> Option<Vec<usize>> {

        if self.dimensions.len() != 1 {
            None
        } else {
            Some(sliding_window_1d(&self.dimensions[0]))
        }
    }

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

    sliding_window(&[*dp]).get(0).unwrap().clone()
}

/// Funciton to create a sliding window over two dimensions.
pub fn sliding_window_2d(x: &DimensionParameters, y: &DimensionParameters) -> Vec<(usize, usize)> {

    sliding_window(&[*x, *y])
        .iter().map(|v| (v[0], v[1])).collect::<Vec<(usize, usize)>>()
}

// -------------------------------------------------------------------------

/// Sliding window of fixed size over a string.
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

/// Create a sliding window of fixed size over a string.
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

/// Sliding window of fixed size over bytes.
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

/// Create a sliding window of fixed size over a string.
pub fn byte_slider<'a>(s: &'a [u8], winlen: usize) -> Option<ByteSlider<'a>> {

    if winlen == 0 {
        return None;
    }

    Some(ByteSlider {
        s: s,
        winlen: winlen,
        pos: 0
    })
}

// -------------------------------------------------------------------------
// TODO tests
