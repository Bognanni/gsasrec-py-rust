use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use ort::execution_providers::CUDAExecutionProvider;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;


#[pyclass]
pub struct Recommender {
    session: Session,
}

#[pymethods]
impl Recommender {
    #[new]
    pub fn new(model_path: &str) -> PyResult<Self> {
        println!("ONNX Runtime initialized.");

        let session = Session::builder()
            .map_err(|e| PyRuntimeError::new_err(format!("Builder error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PyRuntimeError::new_err(format!("Optimization error: {}", e)))?
            .with_execution_providers([
                CUDAExecutionProvider::default().build()
            ])
            .map_err(|e| PyRuntimeError::new_err(format!("Execution provider error: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Recommender { session })
    }

    pub fn get_embeddings(&mut self, padded_batch: Vec<Vec<i64>>) -> PyResult<Vec<f32>> {
        let batch_size = padded_batch.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let max_length = padded_batch[0].len();

        let mut flattened_batch = Vec::with_capacity(batch_size * max_length);
        for sequence in padded_batch {
            flattened_batch.extend(sequence);
        }

        let input_shape = vec![batch_size, max_length];
        let input_tensor = Value::from_array((input_shape, flattened_batch))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let inputs = ort::inputs!["input_seq" => input_tensor];

        let start_time = Instant::now();
        
        let outputs = self.session.run(inputs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            
        let duration = start_time.elapsed();
        let latency_ms = duration.as_secs_f64() * 1000.0;

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("latencies.csv")
        {
            let _ = writeln!(file, "{}", latency_ms);
        }

        let embeddings_tensor = outputs["embedded"].try_extract_tensor::<f32>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(embeddings_tensor.1.to_vec())
    }
}