#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape<const R: usize> {
    pub shape: [usize; R],
}

impl<const R: usize> Shape<R> {
    pub fn new(shape: [usize; R]) -> Shape<R> {
        Shape { shape }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn reshape(&self, shape: [usize; R]) -> Shape<R> {
        Shape { shape }
    }
}

impl Shape<1> {
    pub fn x(&self) -> usize {
        self.shape[0]
    }
}

impl Shape<2> {
    pub fn x(&self) -> usize {
        self.shape[0]
    }

    pub fn y(&self) -> usize {
        self.shape[1]
    }
}

impl Shape<3> {
    pub fn x(&self) -> usize {
        self.shape[0]
    }

    pub fn y(&self) -> usize {
        self.shape[1]
    }

    pub fn z(&self) -> usize {
        self.shape[2]
    }
}
