/// # Errors
/// This module contains all Error enums that can be
/// sent by [quick_math]

///
/// # Matrix Error
/// Enum with errors related to [Matrix] operationErrorss
#[derive(Debug)]
pub enum MatrixError {
    ElementwiseDimensionsMismatch {
        size_1: Vec<usize>,
        size_2: Vec<usize>,
    },
    MatMulDimensionsMismatch {
        size_1: Vec<usize>,
        size_2: Vec<usize>,
    },
    InvalidIndex,
    InvalidReshape {
        numel: usize,
        forcing_into: usize,
    },

    InvalidShape {
        numel: usize,
        forcing_into: Vec<usize>,
    },
}
