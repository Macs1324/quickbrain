use crate::quick_grad::var::Var;

/// # Activation Functions
/// Activation functions are used to restrict the output of a neuron to a certain range
/// This is done to prevent the network from exploding or vanishing gradients
/// The activation function is applied to the weighted sum of the inputs of a neuron
/// The activation function is also used to calculate the error of a neuron
#[derive(Copy, Clone)]
pub enum Activation {
    /// # No Activation
    /// Used to output any value, no restrictions
    NoActivation,
    /// # ReLU
    /// Used to restrict the output of a neuron to a value between 0 and infinity
    /// You can combine this with a layer with no activation to be able to output any value
    ReLU,
    /// # Sigmoid
    /// Used to restrict the output of a neuron to a value between 0 and 1
    Sigmoid,
    /// # Custom Activation
    /// Pass your own activation function f and its derivative d
    /// Warning: Make sure that d is the derivative of f, there are no checks for this
    Custom {
        f: fn(Var) -> Var,
        d: fn(Var) -> Var,
    },
}

impl Activation {
    pub fn get_f(&self) -> fn(Var) -> Var {
        match self {
            Self::NoActivation => |x| x,
            Self::ReLU => __relu,
            Self::Sigmoid => __sigmoid,

            Self::Custom { f, .. } => *f,
        }
    }
}

fn __relu(x: Var) -> Var {
    if x.value() <= 0.0 {
        x.tape().var(0.0)
    } else {
        x.tape().var(x.value())
    }
}

fn __sigmoid(x: Var) -> Var {
    1.0 / (1.0 + (-x).exp())
}
