use crate::quick_grad::{grad_tape::GradTape, var::Var};

use super::{
    errors::TensorError,
    matrix::Matrix,
    shape::Shape,
    tensor::Tensor,
};

pub struct NDArray<T: Copy> {
    pub data: Vec<T>,
    pub shape: Shape,
}

impl NDArray<f64> {
    pub fn zero(shape: Shape) -> NDArray<f64> {
        let data = vec![0.0; shape.shape.iter().product()];
        NDArray { data, shape }
    }

    pub fn one(shape: Shape) -> NDArray<f64> {
        let data = vec![1.0; shape.shape.iter().product()];
        NDArray { data, shape }
    }

    pub fn rand(shape: Shape) -> NDArray<f64> {
        let data = vec![rand::random(); shape.shape.iter().product()];
        NDArray { data, shape }
    }
}

impl NDArray<Var> {
    pub fn zero(tape: &GradTape, shape: Shape) -> NDArray<Var> {
        let data = vec![tape.var(0.0); shape.shape.iter().product()];
        NDArray { data, shape }
    }
}

impl<T: Copy> Tensor<T> for NDArray<T> {
    fn get_data(&self) -> &Vec<T> {
        &self.data
    }

    fn get_data_mut(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    fn map(&self, f: fn(T) -> T) -> Self {
        NDArray {
            data: self.data.iter().copied().map(f).collect(),
            shape: self.shape.clone(),
        }
    }

    fn get_shape(&self) -> Shape {
        self.shape.clone()
    }

    fn cat(&self, _other: &Self, _axis: usize) -> Self {
        todo!("Unimplemented");
    }

    fn reshape(&self, new_shape: Shape) -> Result<NDArray<T>, TensorError> {
        Ok(NDArray {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    fn into_matrix(&self) -> Result<Matrix<T>, TensorError> {
        Matrix::from_data_and_shape(
            self.data.clone(),
            Shape {
                shape: vec![self.shape[0], self.shape[1]],
            },
        )
    }
}
