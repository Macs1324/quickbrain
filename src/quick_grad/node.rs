use super::operation::Operation;

pub struct Node {
    pub value: f64,
    pub grad: Option<f64>,
    pub op: Operation,
    pub parents: Vec<Node>,
    pub children: Vec<Node>,
}
