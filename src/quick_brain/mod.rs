pub mod activation;
pub mod cost;

use crate::{
    quick_grad::{grad::Grad, grad_tape::GradTape, var::Var},
    quick_math::matrix::Matrix as RawMatrix,
};
pub use activation::Activation;

use self::cost::Cost;

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

    /// # Get Parameters
    ///
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
/// use quickbrain::quick_brain::Dense;
/// use quickbrain::quick_brain::Activation;
/// use quickbrain::quick_brain::Sequential;
/// use quickbrain::quick_grad::grad_tape::GradTape;
/// let t = GradTape::new();
/// let mut model = Sequential::new();
/// model.add_layer(Dense::new(&t, 2, 3, Activation::Sigmoid));
/// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
/// ```
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    /// # New
    /// Creates a new Sequential model
    /// ```
    /// use quickbrain::quick_brain::{Sequential, Dense, Activation};
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    /// let mut t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
    /// ```
    pub fn new() -> Sequential {
        Sequential { layers: vec![] }
    }

    /// # Add Layer
    /// Adds a layer to the model
    /// The layer must implement the Layer trait
    /// ```
    /// use quickbrain::quick_brain::Dense;
    /// use quickbrain::quick_brain::Activation;
    /// use quickbrain::quick_brain::Sequential;
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    /// let t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// ```
    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// # Forward
    /// Runs the model on the input
    /// ```
    /// use quickbrain::quick_brain::Dense;
    /// use quickbrain::quick_brain::Activation;
    /// use quickbrain::quick_brain::Sequential;
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    /// use quickbrain::quick_math::matrix::Matrix;
    ///
    /// let t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
    /// let input = Matrix::g_from_array(&t, [[2.0, 1.0]]).transpose(&t);
    /// let output = model.forward(&t, &input);
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
    /// use quickbrain::quick_brain::Dense;
    /// use quickbrain::quick_brain::Activation;
    /// use quickbrain::quick_brain::Sequential;
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    /// let t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
    /// let input_shape = model.get_input_shape(); // [2, 1]
    /// ```
    pub fn get_input_shape(&self) -> Vec<usize> {
        self.layers[0].get_input_shape()
    }

    /// # Get Output Shape
    /// Returns the output shape of the model
    /// ```
    /// use quickbrain::quick_brain::Dense;
    /// use quickbrain::quick_brain::Activation;
    /// use quickbrain::quick_brain::Sequential;
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    /// let t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
    /// let output_shape = model.get_output_shape(); // [1, 1]
    /// ```
    pub fn get_output_shape(&self) -> Vec<usize> {
        self.layers[self.layers.len() - 1].get_output_shape()
    }

    /// # Parameters
    /// Returns a vector of the parameters of the model
    /// ```
    /// use quickbrain::quick_brain::Dense;
    /// use quickbrain::quick_brain::Activation;
    /// use quickbrain::quick_brain::Sequential;
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    ///
    /// let t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
    /// let params = model.parameters();
    /// ```
    pub fn parameters(&mut self) -> Vec<&mut Var> {
        let mut params = vec![];
        for layer in self.layers.iter_mut() {
            for param in layer.parameters() {
                params.push(param);
            }
        }
        params
    }

    /// # Fit
    /// Trains the model on the given dataset
    /// ```
    /// use quickbrain::quick_brain::Dense;
    /// use quickbrain::quick_brain::Activation;
    /// use quickbrain::quick_brain::Sequential;
    /// use quickbrain::quick_grad::grad_tape::GradTape;
    /// use quickbrain::quick_math::matrix::Matrix;
    /// use quickbrain::quick_brain::cost::Cost;
    /// let t = GradTape::new();
    /// let mut model = Sequential::new();
    /// model.add_layer(Dense::new(&t, 2, 3, Activation::ReLU));
    /// model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));
    /// let mut x = Matrix::g_from_array(&t, [[2.0, 1.0]]).transpose(&t);
    /// let mut y = Matrix::g_from_array(&t, [[1.0]]);
    /// model.fit(&t, &mut x, &mut y, 100, 0.1, Cost::MSE);
    /// ```
    pub fn fit(
        &mut self,
        tape: &GradTape,
        x: &mut Matrix,
        y: &mut Matrix,
        epochs: usize,
        learning_rate: f64,
        cost: Cost,
    ) {
        // let pb = ProgressBar::new(epochs as u64);
        for _ in 0..epochs {
            let y_hat = self.forward(tape, x);
            let loss = cost.get_f(y.clone(), y_hat.clone());

            let grad = loss.backward();

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

            tape.clear(things_to_keep);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::quick_math::shape::Shape;

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

        assert_eq!(out_shape, Shape::new([3usize, 1]));
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

        assert_eq!(out_shape, Shape::new([3usize, 10]));
    }

    #[test]
    pub fn linear_problem() {
        let t = GradTape::new();
        let mut model = Sequential::new();
        model.add_layer(Dense::new(&t, 1, 2, Activation::NoActivation));

        let mut x: Matrix = Matrix::g_from_array(&t, [[1.0], [2.0], [3.0]]).transpose(&t);
        let mut y: Matrix =
            Matrix::g_from_array(&t, [[5.0, 11.0], [9.0, 21.0], [13.0, 31.0]]).transpose(&t);

        let try1 = model.forward(&t, &x);
        println!("Try1: {:?}", try1);
        let loss1 = Cost::MSE.get_f(y.clone(), try1.clone());
        println!("Loss1: {}", loss1.value());
        model.fit(&t, &mut x, &mut y, 1000, 0.01, Cost::MSE);
        let try2 = model.forward(&t, &x);
        println!("Try2: {:?}", try2);
        let loss2 = Cost::MSE.get_f(y.clone(), try2.clone());
        println!("Loss2: {}", loss2.value());
        assert!(loss2.value() < loss1.value());
    }

    #[test]
    pub fn xor_problem() {
        let t = GradTape::new();
        let mut model = Sequential::new();
        model.add_layer(Dense::new(&t, 2, 3, Activation::Sigmoid));
        model.add_layer(Dense::new(&t, 3, 1, Activation::Sigmoid));

        let mut x: Matrix =
            Matrix::g_from_array(&t, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
                .transpose(&t);
        let mut y: Matrix = Matrix::g_from_array(&t, [[0.0], [1.0], [1.0], [0.0]]).transpose(&t);

        let try1 = model.forward(&t, &x);
        println!("{:?}", try1.get_shape());
        println!("Try1: {:?}", try1);
        let loss1 = Cost::MSE.get_f(y.clone(), try1.clone());
        println!("Loss1: {}", loss1.value());
        model.fit(&t, &mut x, &mut y, 1000, 0.01, Cost::MSE);
        let try2 = model.forward(&t, &x);
        println!("Try2: {:?}", try2);
        let loss2 = Cost::MSE.get_f(y.clone(), try2.clone());
        println!("Loss2: {}", loss2.value());
        assert!(loss2.value() < loss1.value());
    }
}
