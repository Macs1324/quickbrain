use std::ops::{Add, Div, Mul, Sub};

use super::{grad::Grad, grad_tape::GradTape};

#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t GradTape,
    index: usize,
    value: f64,
}

impl<'t> Add for Var<'t> {
    type Output = Var<'t>;

    fn add(self, other: Var<'t>) -> Var<'t> {
        let tape = self.tape;
        Var {
            tape: self.tape,
            index: tape.push_binary(self.index, other.index, 1.0, 1.0),
            value: self.value + other.value,
        }
    }
}

impl<'t> Sub for Var<'t> {
    type Output = Var<'t>;

    fn sub(self, other: Var<'t>) -> Var<'t> {
        let tape = self.tape;
        tape.push_binary(self.index, other.index, 1.0, -1.0);
        Var {
            tape: self.tape,
            index: tape.push_binary(self.index, other.index, 1.0, -1.0),
            value: self.value - other.value,
        }
    }
}

impl<'t> Mul for Var<'t> {
    type Output = Var<'t>;

    fn mul(self, other: Var<'t>) -> Var<'t> {
        let tape = self.tape;
        Var {
            tape: self.tape,
            index: tape.push_binary(self.index, other.index, other.value, self.value),
            value: self.value * other.value,
        }
    }
}

impl<'t> Div for Var<'t> {
    type Output = Var<'t>;

    fn div(self, other: Var<'t>) -> Var<'t> {
        let tape = self.tape;
        Var {
            tape: self.tape,
            index: tape.push_binary(
                self.index,
                other.index,
                1.0 / other.value,
                -self.value / (other.value * other.value),
            ),
            value: self.value / other.value,
        }
    }
}

impl<'t> Var<'t> {
    pub fn new(tape: &'t GradTape, index: usize, value: f64) -> Self {
        let tape = tape;
        Var { tape, index, value }
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn backward(&self) -> Grad {
        let len = self.tape.nodes.borrow().len();
        let nodes = self.tape.nodes.borrow();

        let mut derivs = vec![0.0; len];
        derivs[self.index] = 1.0;

        for i in (0..len).rev() {
            let node = &nodes[i];
            let deriv = derivs[i];

            for j in 0..2 {
                derivs[node.deps[j]] += deriv * node.weights[j];
            }
        }

        Grad::new(derivs)
    }

    pub fn sin(self) -> Var<'t> {
        let tape = self.tape;
        Var {
            tape: self.tape,
            index: tape.push_unary(self.index, self.value.cos()),
            value: self.value.sin(),
        }
    }

    pub fn cos(self) -> Var<'t> {
        let tape = self.tape;
        Var {
            tape: self.tape,
            index: tape.push_unary(self.index, -self.value.sin()),
            value: self.value.cos(),
        }
    }
}
