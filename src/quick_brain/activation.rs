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
    /// # Tanh
    /// The tanh activation function is a scaled version of the sigmoid function
    Tanh,
    /// # Custom Activation
    /// Pass your own activation function f and its derivative d
    /// Warning: Make sure that d is the derivative of f, there are no checks for this
    Custom { f: fn(f64) -> f64, d : fn(f64) -> f64},
}

impl Activation {
    pub fn get_f(&self) -> fn(f64) -> f64 {
        match self {
            Self::NoActivation => |x| x,
            Self::ReLU => __relu,
            Self::Sigmoid => __sigmoid,
            Self::Tanh => __tanh,

            Self::Custom { f, .. } => *f,
        }
    }

    pub fn get_d(&self) -> fn(f64) -> f64 {
        match self {
            Self::NoActivation => |_x| 1.0,
            Self::ReLU => __relu_d,
            Self::Sigmoid => __sigmoid_d,
            Self::Tanh => __tanh_d,

            Self::Custom { d, .. } => *d,
        }
    }
}

fn __relu(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        x
    }
}
fn __relu_d(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0
    }
}

fn __sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn __sigmoid_d(x: f64) -> f64 {
    let s = __sigmoid(x);
    s * (1.0 - s)
}

fn __tanh(x: f64) -> f64 {
    x.tanh()
}
fn __tanh_d(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}
