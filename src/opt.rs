//! Module for optimization with gradient descent.
//!
//! # Example: Gradient descent
//!
//! The following example minimizes the function f(x) = (x-2)² with gradient
//! descent.
//!
//! ```
//! # extern crate rustml;
//! # extern crate num;
//! use rustml::opt::*;
//! use num::pow;
//!
//! # fn main() {
//! let opts = empty_opts()
//!     .iter(10)     // set the number of iterations to 10
//!     .alpha(0.1)   // set the learning reate
//!     .eps(0.001);  // stopping criterion
//!
//! let r = opt(
//!     &|p| pow(p[0] - 2.0, 2),       // objective to be minimized: (x-2)^2
//!     &|p| vec![2.0 * (p[0] - 2.0)], // derivative
//!     &[4.0],                        // initial parameters
//!     opts                           // optimization options
//! );
//!
//! for (iter, i) in r.fvals.iter().enumerate() {
//!     println!("error after iteration {} was {}", iter + 1, i.1);
//! }
//! println!("solution: {:?}", r.params);
//! assert!(r.params[0] - 2.0 <= 0.3);
//! # }
//! ```
//!
//! See [here](https://github.com/daniel-e/rustml/blob/master/examples/gradient_descent.rs) for
//! another example.
extern crate num;
extern crate rand;

use self::rand::{thread_rng, Rng};

use ops::*;
use regression::*;
use matrix::Matrix;
use opencv::{Window, RgbImage};
use octave::builder;

/// Creates a container that holds the parameters for an optimization algorithm.
#[derive(Copy, Clone)]
pub struct OptParams<T: Clone> {
    /// learning rate
    pub alpha: Option<T>,
    /// number of iterations
    pub iter: Option<usize>,
    /// stopping criterion
    pub eps: Option<T>,
}

impl <T: Clone> OptParams<T> {
    /// Creates a new container with no parameters.
    ///
    /// # Example
    /// 
    /// ```
    /// use rustml::opt::*;
    /// let opts = OptParams::<f64>::new();
    /// assert!(opts.alpha.is_none());
    /// ```
    pub fn new() -> OptParams<T> {
        OptParams {
            alpha: None,
            iter: None,
            eps: None,
        }
    }

    /// Sets the learning rate.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::opt::*;
    /// let opts = empty_opts().alpha(0.2);
    /// assert_eq!(opts.alpha.unwrap(), 0.2);
    /// ```
    pub fn alpha(&self, val: T) -> OptParams<T> {
        OptParams {
            alpha: Some(val),
            iter: self.iter.clone(),
            eps: self.eps.clone(),
        }
    }

    /// Sets the maximum number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::opt::*;
    /// let opts = empty_opts().iter(100);
    /// assert_eq!(opts.iter.unwrap(), 100);
    /// ```
    pub fn iter(&self, val: usize) -> OptParams<T> {
        OptParams {
            alpha: self.alpha.clone(),
            iter: Some(val),
            eps: self.eps.clone(),
        }
    }

    /// Sets the stopping criterion.
    ///
    /// # Example
    ///
    /// ```
    /// use rustml::opt::*;
    /// let opts = empty_opts().eps(0.01);
    /// assert_eq!(opts.eps.unwrap(), 0.01);
    /// ```
    pub fn eps(&self, val: T) -> OptParams<T> {
        OptParams {
            alpha: self.alpha.clone(),
            iter: self.iter.clone(),
            eps: Some(val),
        }
    }
}

/// Returns an empty set of options for optimization algorithms.
pub fn empty_opts() -> OptParams<f64> {
    OptParams::new()
}

/// The result of an optimization.
pub struct OptResult<T> {
    /// TODO
    pub fvals: Vec<(Vec<T>, T)>,
    /// The parameters after the last iteration.
    pub params: Vec<T>,
    /// True if the stopping criterion is fulfilled.
    pub stopped: bool
}


impl <T: Clone + Copy> OptResult<T> {

    /// Creates a matrix from the intermediate parameters and
    /// values of the objective funciton after each iteration.
    pub fn matrix(&self) -> Matrix<T> {

        if self.fvals.len() == 0 {
            return Matrix::new();
        }

        let mut m: Matrix<T> = Matrix::new();
        for &(ref v, f) in self.fvals.iter() {
            let mut x = v.clone();
            x.push(f);
            m.add_row(&x);
        }
        m
    }
}

/// Minimizes an objective using gradient descent.
///
/// The objective `f` is minimized using a standard gradient descent algorithm. The
/// argument `fd` must return the values of the derivatives for each parameter
/// and is executed in each iteration for the current parameters. The argument
/// `init` contains the initial parameters and `opts` contains the options
/// for the gradient descent algorithm.
///
/// <script type="text/javascript"
///   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
/// </script>
/// <script type="text/x-mathjax-config">
///   MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
/// </script>
/// 
/// If $f(\theta_0, \dots, \theta_n)$ is the objective that is to be minimized with
/// the parameters $\theta_0, \dots, \theta_n$ the algorithm works as follows:
///
/// <div style="padding-left: 10px; background: #eeeeee">
/// $[\theta_0, \dots, \theta_n] \leftarrow$ init <br>
/// $\alpha \leftarrow$ opts.alpha <br>
/// $\epsilon \leftarrow$ opts.epsilon <br>
/// <br>
/// for i = 1 to opts.iter do <br>
/// <div style="padding-left:20px"> 
///     $tmp \leftarrow
///         [\theta_0, \dots, \theta_n] - \alpha
///         \left[ \frac{d}{\partial \theta_0} f(\theta_0, \dots, \theta_n)
///          , \dots, 
///          \frac{d}{\partial \theta_n} f(\theta_0, \dots, \theta_n) \right]$ <br>
///
/// if for all $|{tmp}_i - \theta_i| \leq \epsilon \rightarrow$ stop <br>
///
/// $[\theta_0, \dots, \theta_n] \leftarrow tmp$ <br>
/// </div>
/// done <br>
/// </div>
///
/// The vector $\left[ \frac{d}{\partial \theta_0} f(\theta_0, \dots, \theta_n)
///      , \dots, 
///      \frac{d}{\partial \theta_n} f(\theta_0, \dots, \theta_n) \right]$ needs to be
/// returned by `fd`.
///
/// If `alpha` is not specified in `opts` the value 0.1 is used. If the number of
/// iterations is not specified in `opts` the value 1000 is used. If `epsilon` is
/// not specified in `opts` no stopping criterion is checked.
///
/// # Example
/// 
/// ```
/// # extern crate rustml;
/// # extern crate num;
/// use rustml::opt::*;
/// use num::pow;
///
/// # fn main() {
/// // set the number of iterations to 10
/// let opts = empty_opts().iter(10);
///
/// let r = opt(
///     &|p| pow(p[0] - 2.0, 2),       // objective to be minimized: (x-2)^2
///     &|p| vec![2.0 * (p[0] - 2.0)], // derivative
///     &[4.0],                        // initial parameters
///     opts                           // optimization options
/// );
///
/// for (iter, i) in r.fvals.iter().enumerate() {
///     println!("error after iteration {} was {}", iter + 1, i.1);
/// }
/// println!("solution: {:?}", r.params);
/// assert!(r.params[0] - 2.0 <= 0.3);
/// # }
/// ```
pub fn opt<O, D>(f: &O, fd: &D, init: &[f64], opts: OptParams<f64>) -> OptResult<f64>
    where O: Fn(&[f64]) -> f64, D: Fn(&[f64]) -> Vec<f64> {

    let alpha = opts.alpha.unwrap_or(0.1);
    let iter = opts.iter.unwrap_or(1000);
    let eps = opts.eps;

    let mut r = vec![];
    let mut p = init.to_vec();
    let mut stopped = false;

    for _ in (0..iter) {
        let i = p.sub(&fd(&p).mul_scalar(alpha));
        r.push((i.clone(), f(&i)));
        stopped = eps.is_some() && i.iter().zip(p.iter()).all(|(&x, &y)| num::abs(x - y) <= eps.unwrap());
        p = i;
        if stopped {
            break;
        }
    }

    OptResult {
        params: p.to_vec(),
        fvals: r,
        stopped: stopped
    }
}

// TODO duplicated code
pub fn opt_hypothesis(h: &Hypothesis, x: &Matrix<f64>, y: &[f64], opts: OptParams<f64>) -> OptResult<f64> {

    let alpha = opts.alpha.unwrap_or(0.1);
    let iter = opts.iter.unwrap_or(1000);
    let eps = opts.eps;

    let mut r = vec![];
    let mut p = h.params();
    let mut stopped = false;

    let mut hx = Hypothesis::from_params(&p);

    for _ in (0..iter) {
        let d = hx.derivatives(x, y);
        let i = p.sub(&d.mul_scalar(alpha));
        hx = Hypothesis::from_params(&i);
        r.push((i.clone(), hx.error(&x, &y)));
        stopped = eps.is_some() && i.iter().zip(p.iter()).all(|(&x, &y)| num::abs(x - y) <= eps.unwrap());
        p = i;
        if stopped {
            break;
        }
    }

    OptResult {
        params: p.to_vec(),
        fvals: r,
        stopped: stopped
    }
}

/// Plots the learning curve from an optimization result.
///
pub fn plot_learning_curve(r: &OptResult<f64>, w: &Window) -> Result<(String, String), &'static str> {

    let errors = r.fvals.iter().map(|&(_, ref y)| y).cloned().collect::<Vec<f64>>();

    let mut prfx = "/tmp/".to_string();
    prfx.extend(thread_rng().gen_ascii_chars().take(16));

    let script_file = prfx.clone() + ".m";
    let image_file = prfx + ".png";

    let r = builder()
        .add_vector("y = $$", &errors)
        .add("x = 1:size(y, 2)")
        .add("plot(x, y, 'linewidth', 2)")
        .add("grid on")
        .add("title('learning curve')")
        .add("xlabel('iteration')")
        .add("ylabel('error')")
        .add(&("print -r100 -dpng '".to_string() + &image_file + "'"))
        .run(&script_file);

    match r {
        Ok(_) => {
            let img = RgbImage::from_file(&image_file);
            match img {
                Some(i) => {
                    w.show_image(&i);
                    Ok((script_file, image_file))
                },
                _ => Err("Could not load image.")
            }
        },
        _ => Err("Could not run octave.")
    }
}
