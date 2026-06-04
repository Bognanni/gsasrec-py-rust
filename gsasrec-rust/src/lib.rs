pub mod dataset;
pub mod model;
pub mod config;
pub mod transformer;
pub mod eval;
pub mod endpoint;

use pyo3::prelude::*;

// function called when the endpoint.rs is imported in python
#[pymodule]
fn gsasrec_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<endpoint::Recommender>()?;
    m.add_class::<endpoint::CandleRecommender>()?;
    Ok(())
}