use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};

// Import the physics module
mod audio_ops;

/// 2. PYTHON FUNCTION (Bound API)
/// Note the return type: Bound<'py, PyArray1<f32>>
#[pyfunction]
fn load_track<'py>(py: Python<'py>, path: String) -> PyResult<(Bound<'py, PyArray1<f32>>, u32)> {
    
    // 1. Run Rust Physics
    let (samples, sr) = audio_ops::load_and_normalize(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // 2. Transfer to Python
    // In numpy 0.27, into_pyarray(py) returns a 'Bound' object automatically.
    let py_samples = samples.into_pyarray(py);

    Ok((py_samples, sr))
}

/// 3. MODULE DEFINITION (Bound API)
/// Note: 'm' is now &Bound<'_, PyModule>, not &PyModule
#[pymodule]
fn audio_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_track, m)?)?;
    Ok(())
}