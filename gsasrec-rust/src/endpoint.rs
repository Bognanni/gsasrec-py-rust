use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;


fn prepare_sequence(sequence: &[i64], max_length: usize, padding_value: i64) -> Vec<i64> {
    let mut prepared = Vec::new();
    let seq_len = sequence.len();

    if seq_len > max_length {
        prepared.extend_from_slice(&sequence[(seq_len - max_length)..]);
    } else {
        let pad_len = max_length - seq_len;
        prepared.resize(pad_len, padding_value);
        prepared.extend_from_slice(sequence);
    }
    prepared
}

#[pyclass]
pub struct Recommender {
    session: Session,
}

#[pymethods]
impl Recommender {
    #[new]
    pub fn new(model_path: &str) -> PyResult<Self> {
        let session = Session::builder()
            .map_err(|e| PyRuntimeError::new_err(format!("Builder error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PyRuntimeError::new_err(format!("Optimization error: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Recommender { session })
    }

    pub fn get_embeddings(&mut self, batch_history: Vec<Vec<i64>>) -> PyResult<Vec<f32>> {
        let max_length = 200;
        let padding_value = 0;

        let batch_size = batch_history.len();

        let mut flattened_batch = Vec::with_capacity(batch_size * max_length);

        for history in batch_history {
            let padded = prepare_sequence(&history, max_length, padding_value);
            flattened_batch.extend(padded);
        }

        let input_shape = vec![batch_size, max_length];

        let input_tensor = Value::from_array((input_shape, flattened_batch))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let inputs = ort::inputs!["input_seq" => input_tensor];

        let outputs = self.session.run(inputs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let embeddings_tensor = outputs["embedded"].try_extract_tensor::<f32>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let embeddings_slice = embeddings_tensor.1;

        Ok(embeddings_slice.to_vec())
    }
}