use super::tensor::Tensor;

pub struct NDArray<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
}

impl<T> NDArray<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> NDArray<T> {
        NDArray { data, shape }
    }
    pub fn zero(shape: Vec<usize>) -> NDArray<T> {
        let data = vec![0.0; shape.iter().product()];
        NDArray { data, shape }
    }

    pub fn one(shape: Vec<usize>) -> NDArray<T> {
        let data = vec![1.0; shape.iter().product()];
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

    fn reshape(&self, shape: Vec<usize>) -> Self {
        NDArray {
            data: self.data.clone(),
            shape,
        }
    }

    fn into_matrix(&self) -> super::matrix::Matrix<T> {
        super::matrix::Matrix::new(self.data.clone(), self.shape[0], self.shape[1])
    }
}
