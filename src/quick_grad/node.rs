#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub weights: [f64; 2],
    pub deps: [usize; 2],
}
