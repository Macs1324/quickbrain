pub mod quick_brain;
pub mod quick_grad;
pub mod quick_math;

// use indicatif::ProgressBar;
use quick_brain::{Dense, Sequential, cost::Cost};
//
use crate::{
    quick_grad::{grad_tape::GradTape, var::Var},
    quick_math::matrix::Matrix,
};

fn xor_example() {
    let t = GradTape::new();
    let mut model = Sequential::new();
    model.add_layer(Dense::new(&t, 2, 3, quick_brain::Activation::Sigmoid));
    model.add_layer(Dense::new(&t, 3, 1, quick_brain::Activation::Sigmoid));

    let mut x: Matrix<Var> =
        Matrix::g_from_array(&t, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).transpose(&t);
    let mut y: Matrix<Var> = Matrix::g_from_array(&t, [[0.0], [1.0], [1.0], [0.0]]).transpose(&t);

    println!("{:?}", model.forward(&t, &x));
    model.fit(&t, &mut x, &mut y, 10_000, 0.1, Cost::MSE);
    println!("{:?}", model.forward(&t, &x));

    println!("{:?}", model.parameters());
}

fn linear_example() {
    let t = GradTape::new();
    let mut model = Sequential::new();
    model.add_layer(Dense::new(&t, 1, 2, quick_brain::Activation::NoActivation));

    let mut x: Matrix<Var> = Matrix::g_from_array(&t, [[1.0], [2.0], [3.0]]).transpose(&t);
    let mut y: Matrix<Var> =
        Matrix::g_from_array(&t, [[5.0, 11.0], [9.0, 21.0], [13.0, 31.0]]).transpose(&t);

    println!("{:?}", model.forward(&t, &x));
    model.fit(&t, &mut x, &mut y, 10_000, 0.01, Cost::MSE);
    println!(
        "{:?}",
        model.forward(&t, &Matrix::g_from_array(&t, [[4.0]]).transpose(&t))
    );
    println!("{:?}", model.parameters());
}

fn round(x: f64, decimals: i32) -> f64 {
    let factor = 10.0_f64.powi(decimals as i32);
    (x * factor).round() / factor as f64
}
fn main() {
    xor_example();
    linear_example()
}
