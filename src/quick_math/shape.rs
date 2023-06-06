use std::{ops::Index, usize};
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// # Shape
// A struct representing the shape of a tensor or a matrix.
// R : The rank of the tensor
pub struct Shape {
    pub shape: Vec<usize>,
}

impl Shape {
    pub fn new<const R: usize>(shape: [usize; R]) -> Shape {
        Shape {
            shape: shape.iter().copied().collect(),
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn reshape(&self, shape: &Shape) -> Shape {
        Shape {
            shape: shape.shape.clone(),
        }
    }
    pub fn x(&self) -> usize {
        self.shape[0]
    }
    pub fn y(&self) -> usize {
        self.shape[1]
    }
    pub fn z(&self) -> usize {
        self.shape[2]
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.shape[index]
    }
}

impl Iterator for Shape {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.shape.iter().copied().next()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Shape { shape: value }
    }
}
