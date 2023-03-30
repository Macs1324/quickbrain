pub mod activation;
pub mod cost;

use std::time::Instant;

use crate::{
    quick_grad::{grad::Grad, grad_tape::GradTape, var::Var},
    quick_math::matrix::Matrix as RawMatrix,
};
pub use activation::Activation;

type Matrix = RawMatrix<Var>;

/// # Layer
/// A trait that all layers must implement
pub trait Layer {
    /// # Forward
    /// Performs a forward pass on the layer
    fn forward(&self, tape: &GradTape, input: &Matrix) -> Matrix;
    /// # Get Activation
    /// Returns the activation function of the layer
    fn get_activation(&self) -> Activation;

    /// # Get Input Shape
    /// Returns the input shape of the layer
    fn get_input_shape(&self) -> Vec<usize>;
    /// # Get Output Shape
    /// Returns the output shape of the layer
    fn get_output_shape(&self) -> Vec<usize>;

    fn adjust(&mut self, grad: &Grad, learning_rate: f64);

    fn parameters(&mut self) -> Vec<&mut Var>;
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
        tape: &GradTape,
        input_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Dense {
        Dense {
            weight: Matrix::g_rand(tape, output_size, input_size),
            bias: Matrix::g_rand(tape, 1, output_size),
            activation,
        }
    }
}

impl Layer for Dense {
    fn forward(&self, tape: &GradTape, input: &Matrix) -> Matrix {
        (self.weight.g_dot(tape, input).unwrap() + &self.bias.repeat_h(input.get_cols()))
            .map(self.activation.get_f())
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
    fn adjust(&mut self, grad: &Grad, learning_rate: f64) {
        for w in self.weight.get_data_mut() {
            let grad = grad[&w];
            *w.value_mut() -= learning_rate * grad;
        }
        for b in self.bias.get_data_mut() {
            let grad = grad[&b];
            *b.value_mut() -= learning_rate * grad;
        }
    }

    fn parameters(&mut self) -> Vec<&mut Var> {
        // println!("Dense:");
        // println!("weight:\n {:?}", self.weight);
        // println!("bias:\n {:?}", self.bias);
        let mut params = Vec::new();
        for w in self.weight.get_data_mut() {
            params.push(w);
        }
        for b in self.bias.get_data_mut() {
            params.push(b);
        }

        params
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
    pub fn forward(&self, tape: &GradTape, input: &Matrix) -> Matrix {
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

    pub fn parameters(&mut self) -> Vec<&mut Var> {
        let mut params = vec![];
        for layer in self.layers.iter_mut() {
            for param in layer.parameters() {
                params.push(param);
            }
        }
        params
    }

    pub fn fit(
        &mut self,
        tape: &GradTape,
        x: &mut Matrix,
        y: &mut Matrix,
        epochs: usize,
        learning_rate: f64,
    ) {
        for epoch in 0..epochs {
            let y_hat = self.forward(tape, x);
            let loss = (y_hat - y as &Matrix)
                .map(|x| x * x)
                .get_data()
                .iter()
                .copied()
                .reduce(|a, b| a + b)
                .unwrap();
            let start = Instant::now();
            let grad = loss.backward();
            let end = Instant::now() - start;
            if epoch % 1000 == 0 {
                println!("Epoch: {}", epoch);
                println!("Forward pass complete. Loss: {}", loss.value());
                println!("Time to compute gradient: {:?}", end);
            }

            let numof_layers = self.layers.len();
            for i in (0..numof_layers).rev() {
                let layer = &mut self.layers[i];
                layer.adjust(&grad, learning_rate);
            }

            let mut things_to_keep = self.parameters();
            for element in x.get_data_mut() {
                things_to_keep.push(element);
            }
            for element in y.get_data_mut() {
                things_to_keep.push(element);
            }

            // SOMETHING IS WRONG WITH CALCULATING THE GRADS FOR BIASES, WEIGHTS WORK FINE!!!!

            tape.clear(things_to_keep);
        }
        // println!("Loss: {}", loss.value());
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
