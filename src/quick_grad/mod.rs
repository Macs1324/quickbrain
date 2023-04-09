/// # Quick Grad
/// A simple library for automatic differentiation.
///
/// This implementation is inspired by a toy demonstration of AD described in
/// rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
/// A big rework is likely to be expected in the future.
pub mod grad;
pub mod grad_tape;
pub mod node;
mod tests;
pub mod var;
