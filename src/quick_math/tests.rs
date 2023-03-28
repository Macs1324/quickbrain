#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn create_matrix() {
        let _m = Matrix::zero(2, 3);
    }

    #[test]
    fn get_row() {
        let mut m = Matrix::from_array([[1f64, 2f64, 3f64, 4f64], [5f64, 6f64, 7f64, 8f64]]);

        let row: Vec<f64> = m.get_row(0).copied().collect::<Vec<f64>>();

        assert_eq!(row, vec![1f64, 2f64, 3f64, 4f64]);
        *m.get_row_mut(1).nth(1).unwrap() = 100f64;

        let row: Vec<f64> = m.get_row(1).copied().collect::<Vec<f64>>();
        assert_eq!(row, vec![5f64, 100f64, 7f64, 8f64]);
    }

    #[test]
    fn get_col() {
        let mut m = Matrix::from_array([[1f64, 2f64, 3f64, 4f64], [5f64, 6f64, 7f64, 8f64]]);

        let col: Vec<f64> = m.get_col(0).copied().collect::<Vec<f64>>();

        assert_eq!(col, vec![1f64, 5f64]);

        *m.get_col_mut(1).nth(1).unwrap() = 100f64;

        let row: Vec<f64> = m.get_col(1).copied().collect::<Vec<f64>>();
        assert_eq!(row, vec![2f64, 100f64]);
    }

    #[test]
    fn map() {
        let m = Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);

        assert_eq!(
            m.map(|x| x + 2.0f64),
            Matrix::from_array([[3f64, 4f64, 5f64], [6f64, 7f64, 8f64]])
        );
    }

    #[test]
    fn apply() {
        let mut m = Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);

        m.apply(|x| x + 2.0f64);

        assert_eq!(
            m,
            Matrix::from_array([[3f64, 4f64, 5f64], [6f64, 7f64, 8f64]])
        );
    }

    #[test]
    fn reshape() {
        let m = Matrix::from_array([[1f64, 2f64], [3f64, 4f64], [5f64, 6f64]]);

        assert_eq!(
            m.reshape(2, 3).unwrap(),
            Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]])
        )
    }
    #[test]
    fn transpose() {
        let m = Matrix::from_array([[1f64, 2f64], [3f64, 4f64], [5f64, 6f64]]);

        assert_eq!(
            m.transpose(),
            Matrix::from_array([[1f64, 3f64, 5f64], [2f64, 4f64, 6f64]])
        )
    }

    #[test]
    fn dot() {
        let m1 = Matrix::from_array([[1f64, 2f64, 3f64], [4f64, 5f64, 6f64]]);
        let m2 = Matrix::from_array([[1f64, 2f64], [3f64, 4f64], [5f64, 6f64]]);

        assert_eq!(
            m1.dot(&m2).unwrap(),
            Matrix::from_array([[22f64, 28f64], [49f64, 64f64]])
        );
    }
}
