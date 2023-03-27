use std::ops::Add;

pub struct Node {
    pub weights: [f64; 2],
    pub deps: [usize; 2],
}
