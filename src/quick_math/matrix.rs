use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use rand::random;

use crate::quick_grad::{grad_tape::GradTape, var::Var};

use super::traits::{One, Random, Zero};

/// # Struct **Matrix**
/// A classic implementation of the Matrix data structure
///
/// ---
/// Includes
/// - The possibility to create a Matrix structure starting from various other structures
/// - The possibility to manipulate entire matrices as they were individual items
/// - Operator overloading
/// - Matrix math
/// - Safely reshaping and transposing matrices
/// - Trapping you in a simulation
/// ---
/// ### A demonstration
/// ```
/// let m = Matrix::from_array([[1, 2, 3], [4, 5, 6]]);
/// // [1, 2, 3]
/// // [4, 5, 6]
/// let m2 = Matrix::from_array([[1, 2], [3, 4], [5, 6]])
/// // [1, 2]
/// // [3, 4]
/// // [5, 6]
///
/// // Multiplying matrices by a scalar
/// // Multiplying matrices by matrices
/// // Mapping a function to a matrix
/// let r = m.dot(m2 * 2.0).map(|x| x.exp());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    /// The number of rows that the matrix has
    rows: usize,
    /// The number of columns that the matrix has
    cols: usize,
    /// Raw vector for the Data contained by the Matrix
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// # Get Data
    /// Returns a reference to the plain vector holding the raw data of the matrix
    /// Likely not an useful method
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    /// # Get Shape
    /// Returns a [`Vec<usize>`] representing the shape of the Matrix
    pub fn get_shape(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
    }

    /// # Get row slice
    /// Returns a row of the matrix as a slice, it's very fast!
    pub fn get_row_slice(&self, row: usize) -> &[T] {
        &self.data[(row * self.cols)..(row * self.cols + self.cols)]
    }

    /// # Get row slice mut
    /// Returns a mutable reference to a row of the matrix as a slice
    pub fn get_row_slice_mut(&mut self, row: usize) -> &[T] {
        &mut self.data[(row * self.cols)..(row * self.cols + self.cols)]
    }

    /// # Get row
    /// Returns a reference to a given row of the Matrix as an [Iterator]
    pub fn get_row(&self, row: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(row * self.cols).take(self.cols)
    }

    /// # Get row mut
    /// Returns a mutable reference to a given row of the Matrix as an [Iterator]
    pub fn get_row_mut(&mut self, row: usize) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().skip(row * self.cols).take(self.cols)
    }

    /// # Get column
    /// Returns a reference to a given column of the Matrix as an [Iterator]
    pub fn get_col(&self, col: usize) -> impl Iterator<Item = &T> {
        self.data.iter().skip(col).step_by(self.cols)
    }

    /// # Get column mutable
    /// Returns a mutable reference to a given column of the Matrix as an [Iterator]
    pub fn get_col_mut(&mut self, col: usize) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().skip(col).step_by(self.cols)
    }

    /// # Get rows
    /// A simple getter for the number of rows in the Matrix
    pub fn get_rows(&self) -> usize {
        self.rows
    }
    /// # Get cols
    /// A simple getter for the number of columns in the Matrix
    pub fn get_cols(&self) -> usize {
        self.cols
    }

    /// # Dot
    /// Returns the result of a Matrix multiplication operation -> Dot product
    pub fn dot<U: Mul<T>>(&self, other: &Matrix<U>) -> Option<Matrix<T>> {
        if self.cols != other.get_rows() {
            return None;
        }

        let mut m: Matrix<T> = Matrix::zero(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                m[(i, j)] = 0f64;
                m[(i, j)] = vec_dot(
                    self.get_row(i).copied().collect(),
                    other.get_col(j).copied().collect(),
                )
            }
        }

        Some(m)
    }

    /// # Map
    /// Returns a copy of the matrix with a function f applied to it
    pub fn map(&self, f: fn(T) -> T) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().copied().map(f).collect::<Vec<_>>(),
        }
    }

    /// # Apply
    /// Mutates the matrix by applying a function F to each element
    pub fn apply(&mut self, f: fn(T) -> T) {
        for i in &mut self.data {
            *i = f(*i);
        }
    }

    /// # Reshape
    /// Returns [None] if the data doesn't fit the new size
    /// Returns the reshaped matrix otherwise
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Option<Matrix<T>> {
        if (self.rows * self.cols) != (new_rows * new_cols) {
            None
        } else {
            Some(Matrix {
                rows: new_rows,
                cols: new_cols,
                data: self.data.clone(),
            })
        }
    }

    /// # Transpose
    /// Returns a copy of the transposed matrix
    pub fn transpose(&self) -> Matrix<T> {
        let mut r = Matrix::zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r[(j, i)] = self[(i, j)];
            }
        }

        r
    }
}

impl Matrix<f64> {
    /// # Rand
    /// Creates a matrix filled with random values in the range [0, 1)
    /// of a given size
    pub fn rand(rows: usize, cols: usize) -> Matrix<f64> {
        let mut r = Matrix::zero(rows, cols);

        for i in 0..r.get_rows() {
            for j in 0..r.get_cols() {
                r[(i, j)] = random();
            }
        }

        r
    }

    /// # From array
    /// Creates a new [Matrix] from a 2D array of any shape
    /// # WARNING: Uses static dispatch, this is mostly used for a bunch of constants and for
    /// testing, to not push the limits of it or you will end up with a hige executable
    pub fn from_array<const R: usize, const C: usize>(arr: [[f64; C]; R]) -> Matrix<f64> {
        let mut data: Vec<f64> = Vec::new();
        for row in arr {
            for element in row {
                data.push(element);
            }
        }

        Matrix {
            rows: R,
            cols: C,
            data,
        }
    }
    /// # Zero
    /// Creates a new [Matrix] of the given shape and fills it with ZEROS
    pub fn zero(rows: usize, cols: usize) -> Matrix<f64> {
        let mut data: Vec<f64> = Vec::new();
        for _i in 0..rows * cols {
            data.push(0f64);
        }
        Matrix { rows, cols, data }
    }
    /// # One
    /// Creates a new [Matrix] of the given shape and fills it with ONES
    /// Useful for initializing matrices with dummy values that do not give 0 as a multiplication
    /// result
    pub fn one(rows: usize, cols: usize) -> Matrix<f64> {
        let mut data: Vec<f64> = Vec::new();
        for _i in 0..rows * cols {
            data.push(1f64);
        }
        Matrix { rows, cols, data }
    }
    /// # Fill
    /// Mutates the matrix by filling it with the given value
    pub fn fill(&mut self, value: f64) {
        for item in self.data.iter_mut() {
            *item = value;
        }
    }
}

impl<'t> Matrix<Var<'t>> {
    /// # Random
    /// Creates a new [Matrix] of the given shape and fills it with random values
    pub fn random(tape: &GradTape, rows: usize, cols: usize) -> Matrix<Var> {
        let mut data: Vec<Var> = Vec::new();
        for _i in 0..rows * cols {
            data.push(tape.var(random()));
        }
        Matrix { rows, cols, data }
    }

    pub fn from_array<const R: usize, const C: usize>(
        tape: &GradTape,
        arr: [[f64; C]; R],
    ) -> Matrix<T> {
        let mut data: Vec<Var<'t>> = Vec::new();
        for row in arr {
            for element in row {
                data.push(element);
            }
        }

        Matrix {
            rows: R,
            cols: C,
            data,
        }
    }

    pub fn zero(tape: &GradTape, rows: usize, cols: usize) -> Matrix<f64> {
        let mut data: Vec<f64> = Vec::new();
        for _i in 0..rows * cols {
            data.push(tape.var(0.0));
        }
        Matrix { rows, cols, data }
    }

    pub fn one(tape: &GradTape, rows: usize, cols: usize) -> Matrix<f64> {
        let mut data: Vec<f64> = Vec::new();
        for _i in 0..rows * cols {
            data.push(tape.var(1.0));
        }
        Matrix { rows, cols, data }
    }

    pub fn fill(&mut self, tape: &GradTape, value: f64) {
        for i in 0..self.rows * self.cols {
            self.data[i] = tape.var(value);
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

impl<T, U: Add<T>> Add<Matrix<U>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: Matrix<U>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] += other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Add<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] += other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Add<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] += other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] -= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Sub<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] -= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Sub<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] -= other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] *= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Mul<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] *= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Mul<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] *= other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Div<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] /= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Div<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] /= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T> Div<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] /= other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// Operator overloading for matrix references
impl<T> Add<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] += other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Sub<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] -= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] *= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Div<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] /= other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Add<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] += other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Sub<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] -= other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Mul<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] *= other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T> Div<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] /= other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

fn vec_dot(v1: Vec<f64>, v2: Vec<f64>) -> f64 {
    // print!("{:?} . {:?} ", v1, v2);
    let mut r = 0f64;

    let len = v1.len();
    for i in 0..len {
        r += v1[i] * v2[i];
    }
    // println!(" = {}", r);
    r
}
