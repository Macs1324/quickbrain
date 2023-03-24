use super::matrix::Matrix;
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl From<Matrix> for Tensor {
    fn from(value: Matrix) -> Self {
        Tensor {
            data: value.get_data().clone(),
            shape: vec![value.get_rows(), value.get_cols()],
        }
    }
}
