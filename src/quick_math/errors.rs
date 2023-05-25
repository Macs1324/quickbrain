use super::shape::Shape;

/// # Errors
/// This module contains all Error enums that can be
/// sent by [quick_math]

///
/// # Matrix Error
/// Enum with errors related to [Matrix] operationErrorss
#[derive(Debug)]
pub enum TensorError {
    ElementwiseDimensionsMismatch { size_1: Shape, size_2: Shape },
    MatMulDimensionsMismatch { size_1: Shape, size_2: Shape },
    InvalidIndex { indexing: Shape, size: Shape },
    InvalidReshape { numel: usize, forcing_into: usize },

    InvalidShape { numel: usize, forcing_into: Shape },
}
