use std::ops::Index;

use super::var::Var;

pub struct Grad {
    grads: Vec<f64>,
}

impl Grad {
    pub fn new(grads: Vec<f64>) -> Grad {
        Grad { grads }
    }
}

impl<'t> Index<Var<'t>> for Grad {
    type Output = f64;

    fn index(&self, index: Var<'t>) -> &f64 {
        &self.grads[index.index()]
    }
}
