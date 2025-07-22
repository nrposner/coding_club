use pyo3::prelude::*;
use ndarray::linalg::Dot;
use ndarray::{Array2, ArrayBase, Axis, Ix2, LinalgScalar, OwnedRepr};
use std::ops::AddAssign;

// Define a custom type just to make this a bit more readable
type Matrix<A> = ArrayBase<OwnedRepr<A>, Ix2>;

// Define a Rust function for exponentiating a matrix
// This function has never even heard of Python!
pub fn matrix_power_rust<A>(
    matrix: &Matrix<A>,     // <- type definitions like this are MANDATORY in Rust 
    mut exponent: usize     // by default, all objects are immutable. We need to mark them `mut`
) -> Matrix<A>              // to change them
where
    A: LinalgScalar,
    Matrix<A>: Dot<Matrix<A>, Output = Matrix<A>>
{
    let mut result = Array2::eye(matrix.nrows()); // initializing identity matrix
    let mut base = matrix.to_owned();

    // implementing the binary exponentiation algorithm matrices
    // works in O(log(N)) time 
    while exponent > 0 {
        if exponent % 2 == 1 {
            result = result.dot(&base);
        }
        base = base.dot(&base);
        exponent /= 2;
    }
    result
}

// Define a pyfunction: this is a function written in Rust, which is designed to 
// take inputs from and return outputs to Python
#[pyfunction]
pub fn matrix_power(
    // 1. Accept any Python object. A list of lists is expected
    matrix_obj: &Bound<'_, PyAny>,  // <- Remember how type definitions are mandatory? 
    exponent: usize,                // Well, dealing with a language which doesn't have
) -> PyResult<Vec<Vec<f64>>> {      // strong types, we sometimes need a type that says 'I dunno :P'

}

/// A Python module implemented in Rust.
#[pymodule]
fn coding_club(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;
    Ok(())
}
