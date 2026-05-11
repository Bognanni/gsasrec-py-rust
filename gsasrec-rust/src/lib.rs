pub mod dataset;
pub mod model;
pub mod config;
pub mod transformer;
pub mod eval;
pub mod endpoint;

use pyo3::prelude::*;

// function called when the endpoint.rs is imported in python
#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<endpoint::Recommender>()?;
    Ok(())
}