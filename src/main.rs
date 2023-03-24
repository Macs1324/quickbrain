pub mod quick_brain;
pub mod quick_grad;
pub mod quick_math;

use indicatif::ProgressBar;
use quick_brain::{Dense, Sequential};

use crate::quick_math::matrix::Matrix;

fn xor_example() {
    let mut model = Sequential::new();
    model.add_layer(Dense::new(2, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(4, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(4, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(4, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(4, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(4, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(4, 1, quick_brain::Activation::Sigmoid));

    let x = Matrix::from_array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).transpose();
    let y = Matrix::from_array([[0.0], [1.0], [1.0], [0.0]]).transpose();

    model.fit(&x, &y, 500000, 0.1);
    println!("{:?}", model.forward(&x));
}

fn main() {
    xor_example();
}
