pub struct Node {
    pub weights: [f64; 2],
    pub deps: [usize; 2],

    pub alive: bool,
    pub references: usize,
}

impl Node {
    pub fn add_reference(&mut self) {
        self.references += 1;
    }

    pub fn get_reference_count(&self) -> usize {
        self.references
    }
}
