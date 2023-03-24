pub enum Cost {
    MSE,
}

impl Cost {
    pub fn get_f(&self, y: f64, y_hat: f64) -> f64 {
        match self {
            Cost::MSE => (y - y_hat).powi(2),
        }
    }

    pub fn get_d(&self, y: f64, y_hat: f64) -> f64 {
        match self {
            Cost::MSE => 2.0 * (y - y_hat),
        }
    }
}
