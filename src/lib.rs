use pyo3::prelude::*;
use ndarray::linalg::Dot;
use ndarray::{Array2, ArrayBase, Axis, Ix2, LinalgScalar, OwnedRepr};
use std::ops::AddAssign;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn coding_club(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
