use std::ops::Index;

pub struct Grad {
    grads: Vec<f64>,
}

impl Grad {
    pub fn new(grads: Vec<f64>) -> Grad {
        Grad { grads }
    }
}

impl Index for Grad {
    type Output = f64;

    fn index(&self, index: usize) -> &f64 {
        &self.grads[index]
    }
}
