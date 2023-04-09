use crate::quick_grad::var::Var;

use super::Matrix;

/// # Cost
/// An enum containting the different cost functions
pub enum Cost {
    MSE,
}

impl Cost {
    /// # Get F
    /// Returns a pointer to the corresponding function
    pub fn get_f(&self, y: Matrix, y_hat: Matrix) -> Var {
        match self {
            // Cost::MSE => (y_hat - y)
            Cost::MSE => (y_hat - y)
                .map(|x| x * x)
                .get_data()
                .iter()
                .copied()
                .reduce(|a, b| a + b)
                .unwrap(),
        }
    }
}
