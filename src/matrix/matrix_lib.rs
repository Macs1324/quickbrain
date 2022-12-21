
pub struct Matrix {
    rows : usize,
    cols : usize,
    data : Vec<f64>
}

impl Matrix {
    pub fn zero(rows : usize, cols : usize) -> Matrix {
        let mut data : Vec<f64> = Vec::new();
        for i in 0..rows * cols {
            data.push(0f64);
        }
        Matrix {
            rows,
            cols,
            data
        }
    }
    
    pub fn fill(&mut self, value : f64) {
        for item in self.data.iter_mut() {
            *item = value;
        }
    }

    pub fn 
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn create_matrix() {
        let m = Matrix::zero(2, 3);
    }
}
