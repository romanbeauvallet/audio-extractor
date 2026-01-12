use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

// Import the physics module
mod audio_ops;
use audio_ops::SpectralEngine;

/// 2. PYTHON FUNCTION (Bound API)
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
fn debug_spectral_roundtrip<'py>(
    py: Python<'py>,
    path: String,
) -> PyResult<(Bound<'py, PyArray1<f32>>, u32)> {
    // A. Load File
    let (input_samples, sr) = audio_ops::load_and_normalize(&path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // B. Initialize the Spectral Engine
    let engine = SpectralEngine::new(4096, 1024);

    // C. Run the Physics
    // Time -> Frequency
    let spectrogram = engine.stft(&input_samples);

    // Frequency -> Time
    let reconstructed_samples = engine.istft(&spectrogram);

    // D. Return to Python
    // We return the RECONSTRUCTED audio, not the original.
    let py_output = reconstructed_samples.into_pyarray(py);

    Ok((py_output, sr))
}

/// 3. MODULE DEFINITION (Bound API)
#[pymodule]
fn audio_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_track, m)?)?;
    m.add_function(wrap_pyfunction!(debug_spectral_roundtrip, m)?)?;
    Ok(())
}
