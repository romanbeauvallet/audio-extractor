use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

// Import the physics module
mod audio_ops;
use audio_ops::SpectralEngine;

#[pyfunction]
fn load_track<'py>(py: Python<'py>, path: String) -> PyResult<(Bound<'py, PyArray1<f32>>, u32)> {
    // 1. Run Rust Physics
    let (samples, sr) = audio_ops::load_and_normalize(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // 2. Transfer to Python
    let py_samples = samples.into_pyarray(py);
    Ok((py_samples, sr))
}

#[pyfunction]
fn debug_spectral_roundtrip(path: String) -> PyResult<(Vec<f32>, u32)> {
    // A. Load the data (Interleaved Stereo)
    let (samples, sample_rate) = audio_ops::load_and_normalize(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // B. Initialize the Engine
    let engine = SpectralEngine::new(4096, 1024);

    let reconstruction = audio_ops::process_stereo(&engine, &samples);

    Ok((reconstruction, sample_rate))
}

#[pymodule]
fn audio_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_track, m)?)?;
    m.add_function(wrap_pyfunction!(debug_spectral_roundtrip, m)?)?;
    Ok(())
}
