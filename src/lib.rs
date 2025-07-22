use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
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

    // 2. Try to extract the Python object into a Rust nested vector of floats: `Vec<Vec<f64>>`
    // PyO3 will return a PyErr (raising a Python TypeError) if the object
    // doesn't have the right structure (e.g., it's not a list of lists of floats)
    let nested_vecs: Vec<Vec<f64>> = matrix_obj.extract()?; 
                                            //          !^!
                                            // this `?` notation means 'if this returns an error, 
                                            // return it from this function early'
                                            // in this case, it would return a Python TypeError

    // if we succeed, we transform it into a two-dimensional matrix of floats: ndarray::Array2<f64>

    // Get the dimensions from the nested Vecs
    let n_rows = nested_vecs.len();
    if n_rows == 0 {
        // Handle empty matrix case, returning an error early
        return Err(PyValueError::new_err("Matrices cannot be empty".to_string()));
    }
    let n_cols = nested_vecs[0].len();

    // Flatten the nested vectors into a single vector, just like np.flatten()
    let flat_vec: Vec<f64> = nested_vecs.into_iter().flatten().collect();
    //                                 !^!                   !^!
    // What are these doing here? If you want to iterate over something, you need 
    // to be explicit about it! This means 'transform flat_vec into an iterator in place, 
    // flatten it, and then collect it into a new vector of floats' 
    // Because we used .into_iter() instead of .iter(), nested_vecs no longer exists beyond this
    // line!
}

/// A Python module implemented in Rust.
#[pymodule]
fn coding_club(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;
    Ok(())
}
