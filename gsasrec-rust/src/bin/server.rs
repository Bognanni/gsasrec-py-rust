// Commands to start the server based on build context:
// 1. CPU-only build (standard):
//    cargo run --release --bin server -- --device cpu --engine onnx
//
// 2. GPU-enabled build (when enabling --features cuda):
//    cargo run --release --bin server --features cuda -- --device cuda --engine onnx

use actix_web::{web, App, HttpServer, HttpResponse, Responder, middleware::Logger};
use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarBuilder;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

// Conditional import: include ep and ExecutionProvider only if compiling with the "cuda" feature
#[cfg(feature = "cuda")]
use ort::ep::{self, ExecutionProvider};

use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::model::GSASRec;

#[derive(Deserialize)]
struct EmbeddingsRequest {
    batch_sequences: Vec<Vec<u32>>,
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    user_embeddings: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EngineType {
    Candle,
    Onnx,
}

struct Args {
    use_cuda: bool,
    engine: EngineType,
}

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().collect();

        let mut use_cuda = true;
        let mut engine = EngineType::Candle;

        let mut i = 1;
        while i < raw.len() {
            if raw[i] == "--device" && i + 1 < raw.len() {
                use_cuda = raw[i + 1].trim().to_lowercase() == "cuda";
                i += 2;
            } else if raw[i] == "--engine" && i + 1 < raw.len() {
                let eng_str = raw[i + 1].trim().to_lowercase();
                if eng_str == "onnx" {
                    engine = EngineType::Onnx;
                } else if eng_str == "candle" {
                    engine = EngineType::Candle;
                } else {
                    log::warn!("Unknown engine '{}', defaulting to Candle", eng_str);
                }
                i += 2;
            } else {
                i += 1;
            }
        }

        Self { use_cuda, engine }
    }
}

enum ModelBackend {
    Candle(Arc<GSASRec>),
    Onnx(Arc<Mutex<Session>>),
}

struct AppState {
    model: ModelBackend,
    device: CandleDevice,
    sequence_length: usize,
    pad_val: u32,
}

async fn embeddings_endpoint(state: web::Data<AppState>, req: web::Json<EmbeddingsRequest>,) -> impl Responder {
    let sequences = req.batch_sequences.clone();
    let seq_len = state.sequence_length;
    let pad_val = state.pad_val;

    let model_backend = match &state.model {
        ModelBackend::Candle(m) => ModelBackend::Candle(Arc::clone(m)),
        ModelBackend::Onnx(s) => ModelBackend::Onnx(Arc::clone(s)),
    };
    let device = state.device.clone();

    let infer_result = web::block(move || {
        let b = sequences.len();
        if b == 0 {
            return Ok::<Vec<Vec<Vec<f32>>>, String>(vec![]);
        }

        let mut flat: Vec<u32> = Vec::with_capacity(b * seq_len);
        let mut flat_i64: Vec<i64> = Vec::with_capacity(b * seq_len);

        for seq in sequences.iter() {
            let mut inp = seq.clone();
            if inp.len() > seq_len {
                inp = inp[(inp.len() - seq_len)..].to_vec();
            } else if inp.len() < seq_len {
                let diff = seq_len - inp.len();
                let mut padded = vec![pad_val; diff];
                padded.extend_from_slice(&inp);
                inp = padded;
            }
            flat.extend_from_slice(&inp);
            for &val in &inp {
                flat_i64.push(val as i64);
            }
        }

        match model_backend {
            ModelBackend::Candle(model) => {
                let input_tensor = Tensor::from_vec(flat, (b, seq_len), &device)
                    .map_err(|e| format!("Candle tensor creation error: {}", e))?;

                let (seq_emb, _attentions) = model
                    .forward(&input_tensor, false)
                    .map_err(|e| format!("Forward pass error: {}", e))?;

                let embeddings_vec: Vec<Vec<Vec<f32>>> = seq_emb
                    .to_vec3()
                    .map_err(|e| format!("to_vec3 conversion error: {}", e))?;

                Ok(embeddings_vec)
            }
            ModelBackend::Onnx(session_mutex) => {
                let input_shape = vec![b, seq_len];
                let input_tensor = Value::from_array((input_shape, flat_i64))
                    .map_err(|e| format!("ORT Value creation error: {}", e))?;

                let inputs = ort::inputs!["input_seq" => input_tensor];

                let mut session_lock = session_mutex
                    .lock()
                    .map_err(|_| "Critical error: Mutex poisoned during ONNX inference".to_string())?;

                let outputs = session_lock
                    .run(inputs)
                    .map_err(|e| format!("ONNX session execution error: {}", e))?;

                let extracted = outputs["embedded"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("ONNX output extraction error: {}", e))?;

                let shape = &extracted.0;
                let data = &extracted.1;
                
                let hidden_dim = shape[2] as usize;

                let mut result = Vec::with_capacity(b);
                let mut data_idx = 0;

                for _batch_idx in 0..b {
                    let mut seq_vec = Vec::with_capacity(seq_len);
                    for _seq_idx in 0..seq_len {
                        let mut emb_vec = Vec::with_capacity(hidden_dim);
                        for _h_idx in 0..hidden_dim {
                            emb_vec.push(data[data_idx]);
                            data_idx += 1;
                        }
                        seq_vec.push(emb_vec);
                    }
                    result.push(seq_vec);
                }

                Ok(result)
            }
        }
    })
    .await;

    match infer_result {
        Ok(Ok(embs)) => HttpResponse::Ok().json(EmbeddingsResponse {
            user_embeddings: embs,
        }),
        Ok(Err(err)) => {
            log::error!("Inference error: {}", err);
            HttpResponse::InternalServerError().body(err)
        }
        Err(err) => {
            log::error!("Critical error in worker thread: {}", err);
            HttpResponse::InternalServerError().body("Critical server error")
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();

    log::info!("Starting inference server...");
    let args = Args::parse();

    let num_items = 3416;
    let sequence_length = 200;
    let pad_val = num_items as u32 + 1;

    // Candle device initialization
    let device = if args.use_cuda {
        #[cfg(feature = "cuda")]
        {
            CandleDevice::cuda_if_available(0).expect("Failed to initialize CUDA for Candle")
        }
        #[cfg(not(feature = "cuda"))]
        {
            log::warn!("CUDA device requested, but binary was compiled without 'cuda' feature. Falling back to CPU for Candle.");
            CandleDevice::Cpu
        }
    } else {
        CandleDevice::Cpu
    };

    let model_backend = match args.engine {
        EngineType::Candle => {
            let model_path = "model.safetensors";
            log::info!("Loading Candle backend from {}...", model_path);
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                    .expect("Critical error: Failed to read safetensors file")
            };

            let mut config = GsasrecConfig::new("infer", num_items as u32);
            config.sequence_length = sequence_length;

            let model = GSASRec::new(vb, config).expect("Failed to initialize GSASRec model");
            ModelBackend::Candle(Arc::new(model))
        }
        EngineType::Onnx => {
            let model_path = "models/gsasrec-ml1m-step_86064-t_0.75-negs_256-emb_128-dropout_0.5-metric_0.1974453226738962.onnx";
            log::info!("Loading ONNX backend from {}...", model_path);
            
            // Initialize global ONNX Runtime environment
            let _ = ort::init().with_name("gsasrec_inference").commit();

            let mut builder = Session::builder()
                .expect("Failed to create Session builder")
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .expect("Failed to set optimization level")
                .with_intra_threads(1)
                .expect("Failed to set intra_threads")
                .with_inter_threads(1)
                .expect("Failed to set inter_threads");

            // --- CONDITIONAL CUDA INITIALIZATION ---
            #[cfg(feature = "cuda")]
            {
                if args.use_cuda {
                    let cuda = ep::CUDA::default();

                    if cuda.is_available().unwrap_or(false) {
                        cuda.register(&mut builder)
                            .expect("Failed to register CUDA provider");
                        log::info!("ONNX Runtime successfully initialized on GPU (CUDA EP).");
                    } else {
                        log::warn!("Binary includes CUDA support, but runtime hardware or libraries are unavailable. Automatic fallback to CPU.");
                    }
                } else {
                    log::info!("ONNX Runtime initialized on CPU (as requested by --device cpu).");
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                if args.use_cuda {
                    log::warn!("CUDA requested (--device cuda), but server was compiled in CPU-only mode. Running on CPU.");
                } else {
                    log::info!("ONNX Runtime initialized on CPU.");
                }
            }
            // ------------------------------------------

            let session = builder
                .commit_from_file(model_path)
                .expect("Failed to load specified ONNX model");

            ModelBackend::Onnx(Arc::new(Mutex::new(session)))
        }
    };

    let app_state = web::Data::new(AppState {
        model: model_backend,
        device,
        sequence_length,
        pad_val,
    });

    let bind_address = "0.0.0.0:8080";
    log::info!("Server listening at http://{}", bind_address);

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(app_state.clone())
            .route("/get_embeddings/pure_rust", web::post().to(embeddings_endpoint))
    })
    .workers(9)
    .bind(bind_address)?
    .run()
    .await
}