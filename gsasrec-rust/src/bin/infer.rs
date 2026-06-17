/// Infer for the pure Rust Endpoint
///
/// Use:
///   cargo run --bin infer -- \
///     --model      model.safetensors \
///     --input      datasets/ml1m/test/input.txt \
///     --num-items  3416 \
///     [--output         datasets/ml1m/test/output.txt]
///     [--limit          10]
///     [--batch-size     512]
///     [--sequence-length 200]
///     [--embedding-dim  128]
///     [--num-heads      1]
///     [--num-blocks     2]
///     [--dropout        0.5]
///     [--reuse-embeddings]
///     [--no-filter-rated]
///     [--device         cpu|cuda]
///     [--save-results   results.tsv]

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::model::GSASRec;


struct Args {
    model_path:            PathBuf,
    input_path:            PathBuf,
    output_path:           Option<PathBuf>,
    num_items:             u32,
    sequence_length:       usize,
    embedding_dim:         usize,
    num_heads:             usize,
    num_blocks:            usize,
    dropout_rate:          f32,
    reuse_item_embeddings: bool,
    limit:                 usize,
    batch_size:            usize,
    filter_rated:          bool,
    use_cuda:              bool,
    results_path:          Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().collect();

        let mut model_path            = PathBuf::from("model.safetensors");
        let mut input_path            = PathBuf::from("test_input.txt");
        let mut output_path           = None;
        let mut num_items             = 0u32;
        let mut sequence_length       = 200usize;
        let mut embedding_dim         = 128usize;
        let mut num_heads             = 1usize;
        let mut num_blocks            = 2usize;
        let mut dropout_rate          = 0.5f32;
        let mut reuse_item_embeddings = false;
        let mut limit                 = 10usize;
        let mut batch_size            = 512usize;
        let mut filter_rated          = true;
        let mut use_cuda              = false;
        let mut results_path          = None;

        let mut i = 1;
        while i < raw.len() {
            match raw[i].as_str() {
                "--model"           => { model_path      = PathBuf::from(&raw[i+1]); i += 2; }
                "--input"           => { input_path      = PathBuf::from(&raw[i+1]); i += 2; }
                "--output"          => { output_path     = Some(PathBuf::from(&raw[i+1])); i += 2; }
                "--num-items"       => { num_items        = raw[i+1].parse().expect("--num-items: u32"); i += 2; }
                "--sequence-length" => { sequence_length  = raw[i+1].parse().expect("--sequence-length: usize"); i += 2; }
                "--embedding-dim"   => { embedding_dim    = raw[i+1].parse().expect("--embedding-dim: usize"); i += 2; }
                "--num-heads"       => { num_heads        = raw[i+1].parse().expect("--num-heads: usize"); i += 2; }
                "--num-blocks"      => { num_blocks       = raw[i+1].parse().expect("--num-blocks: usize"); i += 2; }
                "--dropout"         => { dropout_rate     = raw[i+1].parse().expect("--dropout: f32"); i += 2; }
                "--reuse-embeddings"=> { reuse_item_embeddings = true; i += 1; }
                "--limit"           => { limit            = raw[i+1].parse().expect("--limit: usize"); i += 2; }
                "--batch-size"      => { batch_size       = raw[i+1].parse().expect("--batch-size: usize"); i += 2; }
                "--no-filter-rated" => { filter_rated     = false; i += 1; }
                "--device"          => { use_cuda = raw[i+1].trim().to_lowercase() == "cuda"; i += 2; }
                "--save-results"    => { results_path = Some(PathBuf::from(&raw[i+1])); i += 2; }
                other => { eprintln!("Arg not recognized: {}", other); std::process::exit(1); }
            }
        }

        if num_items == 0 {
            eprintln!("Error: --num-items is required (e.g., --num-items 3706)");
            std::process::exit(1);
        }

        Self {
            model_path, input_path, output_path,
            num_items, sequence_length, embedding_dim,
            num_heads, num_blocks, dropout_rate,
            reuse_item_embeddings, limit, batch_size,
            filter_rated, use_cuda, results_path,
        }
    }
}


fn read_sequences(path: &PathBuf) -> Vec<Vec<u32>> {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Unable to open {:?}: {}", path, e));
    BufReader::new(file)
        .lines()
        .map(|l| {
            l.unwrap()
            .split_whitespace()
            .filter_map(|s| s.parse::<u32>().ok())
            .collect()
        })
        .collect()
}

fn read_ground_truth(path: &PathBuf) -> Vec<u32> {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Unable to open {:?}: {}", path, e));
    BufReader::new(file)
        .lines()
        .map(|l| l.unwrap().trim().parse::<u32>().unwrap())
        .collect()
}


// Padding — equals to dataset.get_item()
fn prepare_sequence(seq: &[u32], max_length: usize, pad_val: u32) -> (Vec<u32>, HashSet<u32>) {
    let rated: HashSet<u32> = seq.iter().cloned().collect();
    let mut inp = seq.to_vec();

    if inp.len() > max_length {
        inp = inp[(inp.len() - max_length)..].to_vec();
    } else if inp.len() < max_length {
        let diff = max_length - inp.len();
        let mut padded = vec![pad_val; diff];
        padded.extend_from_slice(&inp);
        inp = padded;
    }

    (inp, rated)
}


// Metrics
fn ndcg_at_k(ranked: &[(u32, f32)], relevant: u32, k: usize) -> f64 {
    for (rank, (item, _)) in ranked.iter().take(k).enumerate() {
        if *item == relevant {
            return 1.0 / (rank as f64 + 2.0).log2();
        }
    }
    0.0
}

fn recall_at_k(ranked: &[(u32, f32)], relevant: u32, k: usize) -> f64 {
    if ranked.iter().take(k).any(|(id, _)| *id == relevant) { 1.0 } else { 0.0 }
}


fn main() -> candle_core::Result<()> {
    let args = Args::parse();

    let device = if args.use_cuda {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };
    println!("Device: {:?}", device);

    println!("Loading weights from {:?} ...", args.model_path);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_path.clone()],
            DType::F32,
            &device,
        )?
    };

    let mut config = GsasrecConfig::new("infer", args.num_items);
    config.sequence_length       = args.sequence_length;
    config.embedding_dim         = args.embedding_dim;
    config.num_heads             = args.num_heads;
    config.num_blocks            = args.num_blocks;
    config.dropout_rate          = args.dropout_rate;
    config.reuse_item_embeddings = args.reuse_item_embeddings;
    config.recommendation_limit  = args.limit;

    println!("Config: num_items={} seq_len={} emb_dim={} heads={} blocks={} reuse_emb={}",
        config.num_items, config.sequence_length, config.embedding_dim,
        config.num_heads, config.num_blocks, config.reuse_item_embeddings);

    let model = GSASRec::new(vb, config.clone())?;
    println!("Model loaded\n");

    let all_sequences  = read_sequences(&args.input_path);
    let n_users        = all_sequences.len();
    let ground_truth   = args.output_path.as_ref().map(read_ground_truth);
    let pad_val        = args.num_items + 1;

    println!("Users to process: {}", n_users);

    let mut all_results: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_users);
    let t0 = Instant::now();

    for (batch_idx, chunk) in all_sequences.chunks(args.batch_size).enumerate() {
        let b = chunk.len();
        let mut flat: Vec<u32> = Vec::with_capacity(b * args.sequence_length);
        let mut rated_batch: Vec<HashSet<u32>> = Vec::with_capacity(b);

        for seq in chunk {
            let (padded, rated) = prepare_sequence(seq, args.sequence_length, pad_val);
            flat.extend_from_slice(&padded);
            rated_batch.push(rated);
        }

        let input_tensor = Tensor::from_vec(flat, (b, args.sequence_length), &device)?;
        let rated_opt    = if args.filter_rated { Some(&rated_batch) } else { None };

        let batch_res = model.get_predictions(&input_tensor, args.limit, rated_opt)?;
        all_results.extend(batch_res);

        let n_batches = (n_users + args.batch_size - 1) / args.batch_size;
        if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == n_batches {
            let elapsed = t0.elapsed().as_secs_f32();
            println!("  Batch {}/{} — {:.2}s — {:.0} users/s",
                batch_idx + 1, n_batches, elapsed,
                all_results.len() as f32 / elapsed);
        }
    }

    println!("\nInference in {:.2}s ({:.0} users/s)\n",
        t0.elapsed().as_secs_f32(),
        n_users as f32 / t0.elapsed().as_secs_f32());

    println!("Preview (first 5 users):");
    for (uid, recs) in all_results.iter().take(5).enumerate() {
        let s: Vec<String> = recs.iter()
            .map(|(id, sc)| format!("{}({:.3})", id, sc))
            .collect();
        println!("  user {:>4}: {}", uid, s.join("  "));
    }

    if let Some(ref gt) = ground_truth {
        assert_eq!(gt.len(), n_users,
            "Rows in --output ({}) != users ({})", gt.len(), n_users);

        let ks = [1usize, 10];
        let mut ndcg   = [0.0f64; 3];
        let mut recall = [0.0f64; 3];

        for (recs, &relevant) in all_results.iter().zip(gt.iter()) {
            for (ki, &k) in ks.iter().enumerate() {
                ndcg[ki]   += ndcg_at_k(recs, relevant, k);
                recall[ki] += recall_at_k(recs, relevant, k);
            }
        }

        println!("\n=== Metrics on {} users ===", n_users);
        println!("{:<14} {:>8}", "Metric", "Value");
        println!("{}", "─".repeat(24));
        for (ki, &k) in ks.iter().enumerate() {
            println!("NDCG@{:<9} {:>8.4}", k, ndcg[ki]   / n_users as f64);
            println!("Recall@{:<7} {:>8.4}", k, recall[ki] / n_users as f64);
        }
    }

    if let Some(ref path) = args.results_path {
        let mut f = File::create(path)
            .unwrap_or_else(|e| panic!("Not able to create {:?}: {}", path, e));
        for recs in &all_results {
            let line: Vec<String> = recs.iter()
                .map(|(id, sc)| format!("{}\t{:.6}", id, sc))
                .collect();
            writeln!(f, "{}", line.join("\t")).unwrap();
        }
        println!("\nResults saved in {:?}", path);
    }

    Ok(())
}