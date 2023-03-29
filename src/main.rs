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
    let t = GradTape::new();
    let mut model = Sequential::new();
    model.add_layer(Dense::new(&t, 2, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(&t, 4, 4, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(&t, 4, 1, quick_brain::Activation::Sigmoid));

    let x: Matrix<Var> =
        Matrix::g_from_array(&t, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).transpose(&t);
    let y: Matrix<Var> = Matrix::g_from_array(&t, [[0.0], [1.0], [1.0], [0.0]]).transpose(&t);

    println!("{:?}", model.forward(&t, &x));
    model.fit(&t, &x, &y, 100000, 0.1);
    println!("{:?}", model.forward(&t, &x));
}

fn linear_example() {
    let t = GradTape::new();
    let mut model = Sequential::new();
    model.add_layer(Dense::new(&t, 1, 1, quick_brain::Activation::ReLU));
    // model.add_layer(Dense::new(&t, 3, 3, quick_brain::Activation::ReLU));
    // model.add_layer(Dense::new(&t, 3, 1, quick_brain::Activation::ReLU));

    let x: Matrix<Var> = Matrix::g_from_array(&t, [[0.0]]).transpose(&t);
    let y: Matrix<Var> = Matrix::g_from_array(&t, [[3.0]]).transpose(&t);

    println!("{:?}", model.forward(&t, &x));
    model.fit(&t, &x, &y, 10000, 0.001);
    println!("{:?}", model.forward(&t, &x));
}

fn round(x: f64, decimals: i32) -> f64 {
    let factor = 10.0_f64.powi(decimals as i32);
    (x * factor).round() / factor as f64
}
fn main() {
    xor_example();
    // linear_example()
}
