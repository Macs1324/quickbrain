use std::usize;

use super::shape::Shape;

pub enum DynShapeError {
    FailedResolveMoreThanOneEmptyDimension,
    FailedResolveInvalidNumel,
}

/// # Dynamic Shape
/// A struct used to index / reshape tensors with one (or potentially more than one) unknown
/// dimension.
///
/// This can be compared to reshape targets with -1 in something like numpy
#[derive(Clone)]
pub struct DynShape {
    pub dyn_shape: Vec<Option<usize>>,
}

impl DynShape {
    pub fn new<const R: usize>(dyn_shape: [Option<usize>; R]) -> DynShape {
        DynShape {
            dyn_shape: dyn_shape.iter().copied().collect(),
        }
    }

    // Creates a new [Shape] instance with the empty dimension resolved to account for a given
    // number of elements in the tensor
    pub fn resolve_for_numel(&self, numel: usize) -> Result<Shape, DynShapeError> {
        // Finding the missing dimension
        let mut indexof_empty: Option<usize> = None;
        let mut current_numel = 1;
        for (i, dim) in self.dyn_shape.iter().enumerate() {
            if dim.is_none() {
                if indexof_empty.is_none() {
                    indexof_empty = Some(i);
                } else {
                    return Err(DynShapeError::FailedResolveMoreThanOneEmptyDimension);
                }
            } else {
                current_numel *= dim.unwrap();
            }
        }

        if numel % current_numel != 0 {
            return Err(DynShapeError::FailedResolveInvalidNumel);
        }
        let missing_dim = numel / current_numel;

        let mut shape = self.dyn_shape.clone();
        shape[indexof_empty.unwrap()] = Some(missing_dim);

        Ok(Shape {
            shape: shape.iter().map(|x| x.unwrap()).collect::<Vec<_>>().into(),
        })
    }
}
