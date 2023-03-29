use super::{node::Node, var::Var};
use std::{cell::RefCell, collections::HashMap};

pub struct GradTape {
    pub nodes: RefCell<Vec<Node>>,
    pub constants: RefCell<HashMap<u64, Var>>,
}

impl GradTape {
    pub fn new() -> Self {
        GradTape {
            nodes: RefCell::new(Vec::new()),
            constants: RefCell::new(HashMap::new()),
        }
    }

    pub fn push_unary(&self, dep: usize, weight: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();

        let len = nodes.len();
        nodes.push(Node {
            weights: [weight, 0.0],
            deps: [dep, 0],
        });
        len
    }

    pub fn push_binary(&self, dep1: usize, dep2: usize, weight1: f64, weight2: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();

        let len = nodes.len();
        nodes.push(Node {
            weights: [weight1, weight2],
            deps: [dep1, dep2],
        });
        len
    }
    pub fn var(&self, value: f64) -> Var {
        let len = {
            let mut nodes = self.nodes.borrow_mut();

            let len = nodes.len();
            nodes.push(Node {
                weights: [0.0, 0.0],
                deps: [0, 0],
            });
            len
        };
        Var::new(self, len, value)
    }

    pub fn constant(&self, value: f64) -> Var {
        let id = value.to_bits();
        let mut constants = self.constants.borrow_mut();
        if let Some(var) = constants.get(&id) {
            return var.clone();
        }
        let var = self.var(value);
        constants.insert(id, var.clone());
        var
    }
}
