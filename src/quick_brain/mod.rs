pub mod activation;
pub mod cost;

use crate::{
    quick_grad::{grad_tape::GradTape, var::Var},
    quick_math::matrix::Matrix as RawMatrix,
};
pub use activation::Activation;

type Matrix = RawMatrix<Var>;

/// # Layer
/// A trait that all layers must implement
pub trait Layer {
    /// # Forward
    /// Performs a forward pass on the layer
    fn forward(&self, tape: &mut GradTape, input: &Matrix) -> Matrix;
    /// # Get Activation
    /// Returns the activation function of the layer
    fn get_activation(&self) -> Activation;

    /// # Get Input Shape
    /// Returns the input shape of the layer
    fn get_input_shape(&self) -> Vec<usize>;
    /// # Get Output Shape
    /// Returns the output shape of the layer
    fn get_output_shape(&self) -> Vec<usize>;
}

/// # Dense
/// A fully connected layer
/// ## Fields
/// - weight: The weight matrix of the layer
/// - bias: The bias matrix of the layer
/// - activation: The activation function of the layer
/// ## Methods
/// - new: Creates a new Dense layer
/// - forward: Performs a forward pass on the layer
/// - get_activation: Returns the activation function of the layer
pub struct Dense {
    pub weight: Matrix,
    pub bias: Matrix,
    pub activation: Activation,
}

impl Dense {
    /// # New
    /// Creates a new Dense layer
    /// ## Arguments¨
    /// - input_size: The size of the input
    /// - output_size: The size of the output
    /// - activation: The activation function of the layer
    /// ## Returns
    /// A new Dense layer
    pub fn new(
        tape: &mut GradTape,
        input_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Dense {
        Dense {
            weight: Matrix::g_rand(tape, output_size, input_size),
            bias: Matrix::g_rand(tape, output_size, 1),
            activation,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, tape: &mut GradTape, input: &Matrix) -> Matrix {
        (self.weight.g_dot(tape, input).unwrap()).map(self.activation.get_f())
    }

    fn get_activation(&self) -> Activation {
        self.activation
    }

    fn get_input_shape(&self) -> Vec<usize> {
        vec![self.weight.get_cols(), 1]
    }

    fn get_output_shape(&self) -> Vec<usize> {
        vec![self.weight.get_rows(), 1]
    }
}

/// # Sequential
/// A sequential model
/// ## Example
/// ```
/// let mut model = Sequential::new();
/// model.add_layer(Dense::new(2, 3, Activation::Sigmoid));
/// model.add_layer(Dense::new(3, 1, Activation::Sigmoid));
/// ```
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    /// # New
    /// Creates a new Sequential model
    /// ```
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(2, 3, Activation::Relu));
    /// model.add_layer(Dense::new(3, 1, Activation::Sigmoid));
    /// ```
    pub fn new() -> Sequential {
        Sequential { layers: vec![] }
    }

    /// # Add Layer
    /// Adds a layer to the model
    /// The layer must implement the Layer trait
    /// ```
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(2, 3, Activation::Relu));
    /// ```
    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// # Forward
    /// Runs the model on the input
    /// ```
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(2, 3, Activation::Relu));
    /// model.add_layer(Dense::new(3, 1, Activation::Sigmoid));
    /// let input = Matrix::new(vec![vec![2.0, 1.0]]);
    /// let output = model.forward(&input);
    /// ```
    pub fn forward(&self, tape: &mut GradTape, input: &Matrix) -> Matrix {
        let mut r = input.clone();

        let len = self.layers.len();
        for i in 0..len {
            r = self.layers[i].forward(tape, &r);
        }

        r
    }

    /// # Get Input Shape
    /// Returns the input shape of the model
    /// ```
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(2, 3, Activation::Relu));
    /// model.add_layer(Dense::new(3, 1, Activation::Sigmoid));
    /// let input_shape = model.get_input_shape(); // [2, 1]
    /// ```
    pub fn get_input_shape(&self) -> Vec<usize> {
        self.layers[0].get_input_shape()
    }

    /// # Get Output Shape
    /// Returns the output shape of the model
    /// ```
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(2, 3, Activation::Relu));
    /// model.add_layer(Dense::new(3, 1, Activation::Sigmoid));
    /// let output_shape = model.get_output_shape(); // [1, 1]
    /// ```
    pub fn get_output_shape(&self) -> Vec<usize> {
        self.layers[self.layers.len() - 1].get_output_shape()
    }

    pub fn fit(
        &mut self,
        tape: &mut GradTape,
        x: &Matrix,
        y: &Matrix,
        epochs: usize,
        learning_rate: f64,
    ) {
        let mut loss = tape.var(0.0);
        for epoch in 0..epochs {
            println!("Epoch: {}", epoch);
            let y_hat = self.forward(tape, x);
            // print y and yhat
            // println!("y: {:?}", y);
            // println!("y_hat: {:?}", y_hat);
            // print the shape of yhat
            loss = (y - &y_hat)
                .map(|x| x * x)
                .get_data()
                .iter()
                .copied()
                .reduce(|a, b| a + b)
                .unwrap();
            println!("Forward pass complete. Loss: {}", loss.value());
            let mut error = y - &y_hat;
            error = error.map(|x| x);

            let numof_layers = self.layers.len();
            for i in (0..numof_layers).rev() {
                let layer = &mut self.layers[i];
            }

            for layer in &mut self.layers {}
        }
        println!("Loss: {}", loss.value());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn mlpp_simple() {
        let mut t = GradTape::new();
        let mut m = Sequential::new();
        m.add_layer(Dense::new(&mut t, 2, 3, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 3, 5, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 5, 5, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 5, 3, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 3, 3, Activation::ReLU));

        let input = Matrix::g_rand(&mut t, 2, 1);
        let out_shape = m.forward(&mut t, &input).get_shape();

        assert_eq!(out_shape, vec![3usize, 1]);
    }

    #[test]
    pub fn mlpp_simple_multiple_entries() {
        let mut t = GradTape::new();
        let mut m = Sequential::new();
        m.add_layer(Dense::new(&mut t, 2, 3, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 3, 5, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 5, 5, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 5, 3, Activation::ReLU));
        m.add_layer(Dense::new(&mut t, 3, 3, Activation::ReLU));

        let input = Matrix::g_rand(&mut t, 2, 10);
        let out_shape = m.forward(&mut t, &input).get_shape();

        assert_eq!(out_shape, vec![3usize, 10]);
    }
}
