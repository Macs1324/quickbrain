use super::{node::Node, var::Var};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Formatter},
};

/// # Grad Tape
/// A struct that holds the nodes of the computation graph
/// ## Fields
/// - nodes: The nodes of the computation graph
/// - constants: The constants of the computation graph
/// ## Methods
/// - new: Creates a new GradTape
/// - clear: Clears the tape
/// - push_unary: Pushes a unary node to the tape
/// - push_binary: Pushes a binary node to the tape
/// - var: Creates a new variable
/// - constant: Creates or gets a constant node
pub struct GradTape {
    pub nodes: RefCell<Vec<Node>>,
    pub constants: RefCell<HashMap<u64, Var>>,
}

impl GradTape {
    /// # New
    /// Creates a new GradTape
    pub fn new() -> Self {
        GradTape {
            nodes: RefCell::new(Vec::new()),
            constants: RefCell::new(HashMap::new()),
        }
    }

    /// # Clear
    /// Clears the tape
    /// ## Arguments
    /// - maintain: The variables to maintain
    pub fn clear(&self, maintain: Vec<&mut Var>) {
        self.constants.borrow_mut().clear();
        self.nodes.borrow_mut().clear();

        for var in maintain {
            *var = self.var(var.value());
        }
    }

    /// # Push Unary
    /// Pushes a node with one parent to the tape
    /// ## Arguments
    /// - dep: The dependency of the node
    /// - weight: The weight of the node
    pub fn push_unary(&self, dep: usize, weight: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();

        let len = nodes.len();
        nodes.push(Node {
            weights: [weight, 0.0],
            deps: [dep, 0],
        });
        len
    }

    /// # Push Binary
    /// Pushes a node with two parents to the tape
    /// ## Arguments
    /// - dep1: The first dependency of the node
    /// - dep2: The second dependency of the node
    /// - weight1: The first weight of the node
    /// - weight2: The second weight of the node
    pub fn push_binary(&self, dep1: usize, dep2: usize, weight1: f64, weight2: f64) -> usize {
        let mut nodes = self.nodes.borrow_mut();

        let len = nodes.len();
        nodes.push(Node {
            weights: [weight1, weight2],
            deps: [dep1, dep2],
        });
        len
    }
    /// # Var
    /// Creates a new variable
    /// ## Arguments
    /// - value: The value of the variable
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

    /// # Constant
    /// Creates or gets a constant node
    /// ## Arguments
    /// - value: The value of the constant
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

impl Debug for GradTape {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Tape {{ nodes: {:?} }}", self.nodes.borrow())
    }
}
