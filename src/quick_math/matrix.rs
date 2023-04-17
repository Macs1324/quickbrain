use std::{
    fmt::Debug,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

use rand::random;

use crate::quick_grad::{grad_tape::GradTape, var::Var};

use super::{errors::MatrixError, tensor::Tensor};

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
/// use quickbrain::quick_math::matrix::Matrix;
/// let m : Matrix<f64> = Matrix::from_array([[1., 2., 3.], [4., 5., 6.]]);
/// // [1, 2, 3]
/// // [4, 5, 6]
/// let m2 : Matrix<f64> = Matrix::from_array([[1., 2.], [3., 4.], [5., 6.]]);
/// // [1, 2]
/// // [3, 4]
/// // [5, 6]
///
/// // Multiplying matrices by a scalar
/// // Multiplying matrices by matrices
/// // Mapping a function to a matrix
/// let r = m.dot(&(m2 * 2.0).map(|x| x * x));
/// ```
#[derive(Clone, PartialEq)]
pub struct Matrix<T> {
    /// The number of rows that the matrix has
    rows: usize,
    /// The number of columns that the matrix has
    cols: usize,
    /// Raw vector for the Data contained by the Matrix
    data: Vec<T>,
}

impl<T: Debug> Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut r = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                r.push_str(&format!("{:?} ", self.data[i * self.cols + j]));
            }

            r.push_str("\n");
        }

        write!(f, "{}", r)
    }
}

impl<T: Copy> Matrix<T> {
    pub fn from_data_and_shape(data: Vec<T>, shape: Vec<usize>) -> Result<Self, MatrixError> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(MatrixError::InvalidShape {
                numel: data.len(),
                forcing_into: shape.clone(),
            });
        }

        Ok(Matrix {
            rows: shape[0],
            cols: shape[1],
            data,
        })
    }
    /// # Get Data
    /// Returns a reference to the plain vector holding the raw data of the matrix
    /// Likely not an useful method
    pub fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn get_data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
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
    /// Returns [MatrixError::InvalidReshape] if the data doesn't fit the new size
    /// Returns the reshaped matrix otherwise
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> Result<Matrix<T>, MatrixError> {
        if (self.rows * self.cols) != (new_rows * new_cols) {
            Err(MatrixError::InvalidReshape {
                numel: self.get_rows() * self.get_cols(),
                forcing_into: new_rows * new_cols,
            })
        } else {
            Ok(Matrix {
                rows: new_rows,
                cols: new_cols,
                data: self.data.clone(),
            })
        }
    }

    // # Repeat
    // Replicates the matrix by copying its columns N times
    pub fn repeat_h(&self, times: usize) -> Matrix<T> {
        let mut data = vec![];
        for x in self.data.clone() {
            for _ in 0..times {
                data.push(x);
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols * times,
            data,
        }
    }
}

impl<T: Copy> Tensor<T> for Matrix<T> {
    fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    fn get_data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    fn map(&self, f: fn(T) -> T) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().copied().map(f).collect::<Vec<_>>(),
        }
    }

    fn reshape(&self, shape: Vec<usize>) -> Self {
        Matrix {
            rows: shape[0],
            cols: shape[1],
            data: self.data.clone(),
        }
    }

    fn cat(&self, other: &Self, axis: usize) -> Self {
        if axis == 0 {
            let mut data = self.data.clone();
            data.extend(other.data.clone());
            Matrix {
                rows: self.rows + other.rows,
                cols: self.cols,
                data,
            }
        } else {
            let mut data = vec![];
            for i in 0..self.rows {
                data.extend(self.get_row_slice(i).to_vec());
                data.extend(other.get_row_slice(i).to_vec());
            }
            Matrix {
                rows: self.rows,
                cols: self.cols + other.cols,
                data,
            }
        }
    }

    fn into_matrix(&self) -> Matrix<T> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }

    fn get_shape(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
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
    /// # Dot
    /// Returns the result of a Matrix multiplication operation -> Dot product
    pub fn dot(&self, other: &Matrix<f64>) -> Result<Matrix<f64>, MatrixError> {
        if self.cols != other.get_rows() {
            return Err(MatrixError::MatMulDimensionsMismatch {
                size_1: self.get_shape(),
                size_2: other.get_shape(),
            });
        }

        let mut m: Matrix<f64> = Matrix::zero(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                m[(i, j)] = 0f64;
                m[(i, j)] = vec_dot(
                    self.get_row(i).copied().collect(),
                    other.get_col(j).copied().collect(),
                )
            }
        }

        Ok(m)
    }
    /// # Transpose
    /// Returns a copy of the transposed matrix
    pub fn transpose(&self) -> Matrix<f64> {
        let mut r = Matrix::zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r[(j, i)] = self[(i, j)];
            }
        }

        r
    }
}

impl Matrix<Var> {
    /// # Random
    /// Creates a new [Matrix] of the given shape and fills it with random values
    pub fn g_rand(tape: &GradTape, rows: usize, cols: usize) -> Matrix<Var> {
        let mut data: Vec<Var> = Vec::new();
        for _i in 0..rows * cols {
            data.push(tape.var(random()));
        }
        Matrix { rows, cols, data }
    }

    pub fn apply_to_value(&mut self, f: fn(x: Var) -> f64) {
        for i in &mut self.data {
            *i.value_mut() = f(*i);
        }
    }

    pub fn g_from_array<const R: usize, const C: usize>(
        tape: &GradTape,
        arr: [[f64; C]; R],
    ) -> Matrix<Var> {
        let mut data: Vec<Var> = Vec::new();
        for row in arr {
            for element in row {
                data.push(tape.var(element));
            }
        }

        Matrix {
            rows: R,
            cols: C,
            data,
        }
    }

    /// # G Zero
    /// Creates a new [Matrix] of the given shape and fills it with ZERO [Var]s
    pub fn g_zero(tape: &GradTape, rows: usize, cols: usize) -> Matrix<Var> {
        let mut data: Vec<Var> = Vec::new();
        for _i in 0..rows * cols {
            data.push(tape.var(0.0));
        }
        Matrix { rows, cols, data }
    }

    /// # G One
    /// Creates a new [Matrix] of the given shape and fills it with ONE [Var]s
    pub fn g_one(tape: &GradTape, rows: usize, cols: usize) -> Matrix<Var> {
        let mut data: Vec<Var> = Vec::new();
        for _i in 0..rows * cols {
            data.push(tape.var(1.0));
        }
        Matrix { rows, cols, data }
    }

    /// # G Fill
    /// Mutates the matrix by filling it with the given value
    pub fn g_fill(&mut self, tape: &GradTape, value: f64) {
        for i in 0..self.rows * self.cols {
            self.data[i] = tape.var(value);
        }
    }

    /// # Dot
    /// Returns the result of a Matrix multiplication operation -> Dot product
    pub fn g_dot(&self, tape: &GradTape, other: &Matrix<Var>) -> Result<Matrix<Var>, MatrixError> {
        if self.cols != other.get_rows() {
            return Err(MatrixError::MatMulDimensionsMismatch {
                size_1: self.get_shape(),
                size_2: other.get_shape(),
            });
        }

        let mut m: Matrix<Var> = Matrix::g_zero(tape, self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                // m[(i, j)] = 0f64;
                m[(i, j)] = g_vec_dot(
                    tape,
                    self.get_row(i).copied().collect(),
                    other.get_col(j).copied().collect(),
                )
            }
        }

        Ok(m)
    }
    /// # Transpose
    /// Returns a copy of the transposed matrix
    pub fn transpose(&self, tape: &GradTape) -> Matrix<Var> {
        let mut r = Matrix::g_zero(tape, self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r[(j, i)] = self[(i, j)];
            }
        }

        r
    }

    /// # Value
    /// Returns a copy of the matrix with all the variables replaced with their values
    pub fn value(&self) -> Matrix<f64> {
        let mut r = Matrix::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r[(i, j)] = self[(i, j)].value();
            }
        }

        r
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

impl<T: Copy + Clone + Add<U, Output = T>, U: Copy + Add<T>> Add<Matrix<U>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: Matrix<U>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] + other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Add<Output = T>> Add<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] + other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Add<f64, Output = T>> Add<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] + other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Sub<Output = T>> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] - other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Sub<Output = T>> Sub<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] - other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Sub<f64, Output = T>> Sub<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] - other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Mul<Output = T>> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] * other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Mul<Output = T>> Mul<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] * other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Mul<f64, Output = T>> Mul<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] * other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Div<Output = T>> Div<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] / other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Div<Output = T>> Div<&Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] / other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}
impl<T: Copy + Clone + Div<f64, Output = T>> Div<f64> for Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] / other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// Operator overloading for matrix references
impl<T: Copy + Clone + Add<Output = T>> Add<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] + other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Add<Output = T>> Sub<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] + other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Mul<Output = T>> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] * other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Div<Output = T>> Div<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: &Matrix<T>) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] / other.data[i];
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Add<f64, Output = T>> Add<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] + other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Sub<f64, Output = T>> Sub<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] - other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Mul<f64, Output = T>> Mul<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] * other;
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl<T: Copy + Clone + Div<f64, Output = T>> Div<f64> for &Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: f64) -> Matrix<T> {
        let mut data = self.data.clone();
        for i in 0..data.len() {
            data[i] = data[i] / other;
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
    let mut r = 0.0;

    let len = v1.len();

    for i in 0..len {
        r = r + v1[i] * v2[i];
    }
    // println!("= {}", r);
    r
}

fn g_vec_dot(tape: &GradTape, v1: Vec<Var>, v2: Vec<Var>) -> Var {
    // print!("{:?} . {:?} ", v1, v2);
    let mut r = tape.var(0.0);

    let len = v1.len();
    for i in 0..len {
        r = r + v1[i] * v2[i];
    }
    // println!(" = {}", r);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn create_matrix() {
        let _m = Matrix::zero(2, 3);
    }

    #[test]
    fn get_row() {
        let mut m = Matrix::from_array([[1f64, 2f64, 3f64, 4f64], [5f64, 6f64, 7f64, 8f64]]);

        let row: Vec<f64> = m.get_row(0).copied().collect::<Vec<f64>>();

        assert_eq!(row, vec![1f64, 2f64, 3f64, 4f64]);
        *m.get_row_mut(1).nth(1).unwrap() = 100f64;

        let row: Vec<f64> = m.get_row(1).copied().collect::<Vec<f64>>();
        assert_eq!(row, vec![5f64, 100f64, 7f64, 8f64]);
    }

    #[test]
    fn get_col() {
        let mut m = Matrix::from_array([[1f64, 2f64, 3f64, 4f64], [5f64, 6f64, 7f64, 8f64]]);

        let col: Vec<f64> = m.get_col(0).copied().collect::<Vec<f64>>();

        assert_eq!(col, vec![1f64, 5f64]);

        *m.get_col_mut(1).nth(1).unwrap() = 100f64;

        let row: Vec<f64> = m.get_col(1).copied().collect::<Vec<f64>>();
        assert_eq!(row, vec![2f64, 100f64]);
    }

    #[test]
    fn map() {
        let m = Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);

        assert_eq!(
            m.map(|x| x + 2.0f64),
            Matrix::from_array([[3f64, 4f64, 5f64], [6f64, 7f64, 8f64]])
        );
    }

    #[test]
    fn apply() {
        let mut m = Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);

        m.apply(|x| x + 2.0f64);

        assert_eq!(
            m,
            Matrix::from_array([[3f64, 4f64, 5f64], [6f64, 7f64, 8f64]])
        );
    }

    #[test]
    fn reshape() {
        let m = Matrix::from_array([[1f64, 2f64], [3f64, 4f64], [5f64, 6f64]]);

        assert_eq!(
            m.reshape(2, 3).unwrap(),
            Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]])
        )
    }
    #[test]
    fn transpose() {
        let m = Matrix::from_array([[1f64, 2f64], [3f64, 4f64], [5f64, 6f64]]);

        assert_eq!(
            m.transpose(),
            Matrix::from_array([[1f64, 3f64, 5f64], [2f64, 4f64, 6f64]])
        )
    }

    #[test]
    fn dot() {
        let m1 = Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);
        let m2 = Matrix::from_array([[1f64, 2f64], [3f64, 4f64], [5f64, 6f64]]);

        assert_eq!(
            m1.dot(&m2).unwrap(),
            Matrix::from_array([[22f64, 28f64], [49f64, 64f64]])
        );
    }

    #[test]
    fn basic_matrix_differentiation() {
        let t = GradTape::new();
        let mut m1 = Matrix::g_from_array(&t, [[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);
        let mut m2 = Matrix::g_from_array(&t, [[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);

        let m3 = &m1 * &m2;

        let grad = m3[(1, 2)].backward();

        assert_eq!(grad[&m2[(1, 2)]], m1[(1, 2)].value());

        t.clear({
            let mut r = Vec::new();
            for x in m1.get_data_mut() {
                r.push(x);
            }
            for x in m2.get_data_mut() {
                r.push(x);
            }

            r
        });
        let m3 = &m1 * &m2;
        let grad = m3[(1, 2)].backward();
        assert_eq!(grad[&m2[(1, 2)]], m1[(1, 2)].value());
    }
}
