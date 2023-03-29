use std::{
    cell::RefCell,
    fmt::{Debug, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use super::{grad::Grad, grad_tape::GradTape};

#[derive(Clone, Copy)]
pub struct Var {
    tape: *const GradTape, // This has to be refactored!
    // It defies all the purposes of Rust, and is currently here just because
    // of ergonomics.
    // At least wrap it into something in the future
    index: usize,
    value: f64,
}

impl Debug for Var {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl Add for Var {
    type Output = Var;

    fn add(self, other: Var) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Cannot get the tape pointer")
                .push_binary(self.index, other.index, 1.0, 1.0)
        };
        Var {
            tape: self.tape,
            index,
            value: self.value + other.value,
        }
    }
}

impl Sub for Var {
    type Output = Var;

    fn sub(self, other: Var) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get the tape pointer")
                .push_binary(self.index, other.index, 1.0, -1.0)
        };
        Var {
            tape: self.tape,
            index,
            value: self.value - other.value,
        }
    }
}

impl Mul for Var {
    type Output = Var;

    fn mul(self, other: Var) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get the tape pointer")
                .push_binary(self.index, other.index, other.value, self.value)
        };
        Var {
            tape: self.tape,
            index,
            value: self.value * other.value,
        }
    }
}

impl Div for Var {
    type Output = Var;

    fn div(self, other: Var) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get the tape pointer")
                .push_binary(
                    self.index,
                    other.index,
                    1.0 / other.value,
                    -self.value / (other.value * other.value),
                )
        };
        Var {
            tape: self.tape,
            index,
            value: self.value / other.value,
        }
    }
}

impl Neg for Var {
    type Output = Var;

    fn neg(self) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get the tape pointer")
                .push_unary(self.index, -1.0)
        };
        Var {
            tape: self.tape,
            index,
            value: -self.value,
        }
    }
}

impl Add<f64> for Var {
    type Output = Var;

    fn add(self, other: f64) -> Var {
        self + self.tape().constant(other)
    }
}

impl Sub<f64> for Var {
    type Output = Var;

    fn sub(self, other: f64) -> Var {
        self - self.tape().constant(other)
    }
}

impl Mul<f64> for Var {
    type Output = Var;

    fn mul(self, other: f64) -> Var {
        self * self.tape().constant(other)
    }
}

impl Div<f64> for Var {
    type Output = Var;

    fn div(self, other: f64) -> Var {
        self / self.tape().constant(other)
    }
}

impl Add<Var> for f64 {
    type Output = Var;

    fn add(self, other: Var) -> Var {
        other + self
    }
}

impl Sub<Var> for f64 {
    type Output = Var;

    fn sub(self, other: Var) -> Var {
        other.tape().constant(self) - other
    }
}

impl Mul<Var> for f64 {
    type Output = Var;

    fn mul(self, other: Var) -> Var {
        other * self
    }
}

impl Div<Var> for f64 {
    type Output = Var;

    fn div(self, other: Var) -> Var {
        other.tape().constant(self) / other
    }
}

impl Var {
    pub fn new(tape: &GradTape, index: usize, value: f64) -> Self {
        Var { tape, index, value }
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn tape(&self) -> &GradTape {
        unsafe { self.tape.as_ref().expect("Failed to get the tape pointer") }
    }

    pub fn backward(&self) -> Grad {
        unsafe {
            let len = self
                .tape
                .as_ref()
                .expect("Failed to get tape poinetr")
                .nodes
                .borrow()
                .len();
            let nodes = self
                .tape
                .as_ref()
                .expect("Failed to get tape pointer")
                .nodes
                .borrow();

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
    }

    pub fn sin(self) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get tape pointer")
                .push_unary(self.index, self.value.cos())
        };
        Var {
            tape: self.tape,
            index,
            value: self.value.sin(),
        }
    }

    pub fn cos(self) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get tape pointer")
                .push_unary(self.index, -self.value.sin())
        };
        Var {
            tape: self.tape,
            index,
            value: self.value.cos(),
        }
    }

    pub fn tan(self) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get tape pointer")
                .push_unary(self.index, 1.0 / (self.value.cos() * self.value.cos()))
        };
        Var {
            tape: self.tape,
            index,
            value: self.value.tan(),
        }
    }

    pub fn exp(self) -> Var {
        let index = unsafe {
            self.tape
                .as_ref()
                .expect("Failed to get tape pointer")
                .push_unary(self.index, self.value.exp())
        };
        Var {
            tape: self.tape,
            index,
            value: self.value.exp(),
        }
    }
}
