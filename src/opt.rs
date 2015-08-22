//! Module for optimization.

extern crate num;

use ops::*;

/// Creates a container that holds the parameters for an optimization algorithm.
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
    /// let opts = OptParams::new();
    /// assert!(opts.alphs.is_none());
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
    /// let opts = OptParams::new().alpha(0.2);
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
    /// let opts = OptParams::new().iter(100);
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
    /// let opts = OptParams::new().eps(0.01);
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

pub struct OptResult<T> {
    /// The values of the objective function after each iteration.
    pub fvals: Vec<T>,
    /// The parameters after the last iteration.
    pub params: Vec<T>,
    /// True if the stopping criterion is fulfilled.
    pub stopped: bool
}

/// Minimizes an objective using gradient descent.
///
/// The objective `f` is minimized using a standard gradient descent algorithm. The
/// argument `fd` must return the values of the derivitives for each parameter
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
/// $[\theta_0, \dots, \theta_n] \leftarrow init$ <br>
/// $\alpha \leftarrow$ opts.alpha <br>
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
pub fn opt<O, D>(f: O, fd: D, init: &[f64], opts: OptParams<f64>) -> OptResult<f64>
    where O: Fn(&[f64]) -> f64, D: Fn(&[f64]) -> Vec<f64> {

    let alpha = opts.alpha.unwrap_or(0.1);
    let iter = opts.iter.unwrap_or(1000);
    let eps = opts.eps;

    let mut r = vec![];
    let mut p = init.to_vec();
    let mut stopped = false;

    for _ in (0..iter) {
        let i = p.sub(&fd(&p).mul_scalar(alpha));
        r.push(f(&i));
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


