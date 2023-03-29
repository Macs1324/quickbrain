pub mod quick_brain;
pub mod quick_grad;
pub mod quick_math;

// use indicatif::ProgressBar;
use quick_brain::{Dense, Sequential};
//
use crate::{
    quick_grad::{grad_tape::GradTape, var::Var},
    quick_math::matrix::Matrix,
};

fn xor_example() {
    let mut t = GradTape::new();
    let mut model = Sequential::new();
    model.add_layer(Dense::new(&t, 2, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::ReLU));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(&t, 4, 1, quick_brain::Activation::Sigmoid));

    let x: Matrix<Var> =
        Matrix::g_from_array(&t, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).transpose(&t);
    let y: Matrix<Var> = Matrix::g_from_array(&t, [[0.0], [1.0], [1.0], [0.0]]).transpose(&t);

    model.fit(&t, &x, &y, 500000, 1.0);
    println!("{:?}", model.forward(&t, &x));
}

fn main() {
    xor_example();
}
