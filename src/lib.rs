use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::linalg::Dot;
use ndarray::{Array2, ArrayBase, Ix2, LinalgScalar, OwnedRepr};

type Matrix<A> = ArrayBase<OwnedRepr<A>, Ix2>;

pub fn matrix_power_rust<A>(
    matrix: &Matrix<A>,     
    mut exponent: usize     
) -> Matrix<A>             
where
    A: LinalgScalar,
    Matrix<A>: Dot<Matrix<A>, Output = Matrix<A>>
{
    let mut result = Array2::eye(matrix.nrows());
    let mut base = matrix.to_owned();
    while exponent > 0 {
        if exponent % 2 == 1 { result = result.dot(&base); }
        base = base.dot(&base);
        exponent /= 2;
    }
    result
}

#[pyfunction]
pub fn matrix_power(
    matrix_obj: &Bound<'_, PyAny>,  
    exponent: usize,               
) -> PyResult<Vec<Vec<f64>>> {     
    let nested_vecs: Vec<Vec<f64>> = matrix_obj.extract()?; 

    let n_rows = nested_vecs.len();
    if n_rows == 0 { return Err(PyValueError::new_err("Matrices cannot be empty".to_string())); }
    let n_cols = nested_vecs[0].len();

    let flat_vec: Vec<f64> = nested_vecs.into_iter().flatten().collect();
    
    let rust_matrix = Array2::from_shape_vec((n_rows, n_cols), flat_vec)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let result_matrix = matrix_power_rust(&rust_matrix, exponent);

    let result_vecs: Vec<Vec<f64>> = result_matrix.rows().into_iter()
        .map(|row| row.to_vec()) .collect();

    Ok(result_vecs)
}

/// A Python module implemented in Rust.
#[pymodule]
fn coding_club(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matrix_power, m)?)?;
    Ok(())
}
