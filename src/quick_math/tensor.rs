use super::{errors::TensorError, matrix::Matrix, shape::Shape};

/// # Tensor Trait
/// This trait defines the basic operations that a tensor should have.
/// A tensor is a generalization of a matrix
/// It should add interoperability between matrices and ndarrays
pub trait Tensor<T: Copy> {
    /// # Get Data
    /// Returns a reference to the data of the tensor
    fn get_data(&self) -> &Vec<T>;
    /// # Get Data Mut
    /// Returns a mutable reference to the data of the tensor
    fn get_data_mut(&mut self) -> &mut Vec<T>;
    /// # Map
    /// Returns a new tensor with the function applied to each element
    fn map(&self, f: fn(T) -> T) -> Self;
    /// # Reshape
    /// Returns a new tensor with the shape specified
    /// The number of elements must be the same
    fn reshape(&self, new_shape: Shape) -> Result<Self, TensorError>
    where
        Self: Sized;
    /// # Cat
    /// Returns a new tensor with the other tensor concatenated
    fn cat(&self, other: &Self, axis: usize) -> Self;
    /// # Into Matrix
    /// Returns a matrix with the data of the tensor
    fn into_matrix(&self) -> Result<Matrix<T>, TensorError>;
    /// # Get Shape
    /// Returns the shape of the tensor
    fn get_shape(&self) -> Shape;
    /// # Index
    fn get(&self, index: Shape) -> Result<Self, TensorError>
    where
        Self: Sized;

    fn apply(&mut self, f: fn(T) -> T) {
        let data = self.get_data_mut();
        for i in 0..data.len() {
            data[i] = f(data[i]);
        }
    }
}
