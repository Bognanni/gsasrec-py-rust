/// Binario di inferenza per gSASRec.
///
/// Uso:
///   cargo run --bin infer -- \
///     --model      model.safetensors \
///     --dataset    data/ml-1m \
///     --input      data/ml-1m/test_input.txt \
///     --output     data/ml-1m/test_output.txt \
///     --num-items  3706 \
///     --limit      10 \
///     [--no-filter-rated] \
///     [--device cpu|cuda]
///
/// Il binario:
///   1. Carica i pesi dal file .safetensors
///   2. Legge le sequenze dal file di input
///   3. Esegue l'inferenza a batch
///   4. Stampa le top-K raccomandazioni per ogni utente
///   5. (opzionale) calcola NDCG@10 e Recall@10 se viene fornito --output

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;

// Importa i moduli locali del crate
use gsasrec_rust::config::GsasrecConfig;
use gsasrec_rust::model::GSASRec;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    model_path:      PathBuf,
    input_path:      PathBuf,
    output_path:     Option<PathBuf>,
    num_items:       u32,
    sequence_length: usize,
    embedding_dim:   usize,
    num_heads:       usize,
    num_blocks:      usize,
    dropout_rate:    f32,
    reuse_item_embeddings: bool,
    limit:           usize,
    batch_size:      usize,
    filter_rated:    bool,
    use_cuda:        bool,
    results_path:    Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let raw: Vec<String> = std::env::args().collect();
        let mut model_path      = PathBuf::from("model.safetensors");
        let mut input_path      = PathBuf::from("test_input.txt");
        let mut output_path     = None;
        let mut num_items       = 0u32;
        let mut sequence_length = 200;
        let mut embedding_dim   = 128;
        let mut num_heads       = 1;
        let mut num_blocks      = 2;
        let mut dropout_rate    = 0.5f32;
        let mut reuse_item_embeddings = false;
        let mut limit           = 10;
        let mut batch_size      = 512;
        let mut filter_rated    = true;
        let mut use_cuda        = false;
        let mut results_path    = None;

        let mut i = 1;
        while i < raw.len() {
            match raw[i].as_str() {
                "--model"          => { model_path = PathBuf::from(&raw[i + 1]); i += 2; }
                "--input"          => { input_path = PathBuf::from(&raw[i + 1]); i += 2; }
                "--output"         => { output_path = Some(PathBuf::from(&raw[i + 1])); i += 2; }
                "--num-items"      => { num_items = raw[i + 1].parse().expect("--num-items deve essere u32"); i += 2; }
                "--sequence-length"=> { sequence_length = raw[i + 1].parse().expect("--sequence-length deve essere usize"); i += 2; }
                "--embedding-dim"  => { embedding_dim = raw[i + 1].parse().expect("--embedding-dim deve essere usize"); i += 2; }
                "--num-heads"      => { num_heads = raw[i + 1].parse().expect("--num-heads deve essere usize"); i += 2; }
                "--num-blocks"     => { num_blocks = raw[i + 1].parse().expect("--num-blocks deve essere usize"); i += 2; }
                "--dropout"        => { dropout_rate = raw[i + 1].parse().expect("--dropout deve essere f32"); i += 2; }
                "--reuse-embeddings" => { reuse_item_embeddings = true; i += 1; }
                "--limit"          => { limit = raw[i + 1].parse().expect("--limit deve essere usize"); i += 2; }
                "--batch-size"     => { batch_size = raw[i + 1].parse().expect("--batch-size deve essere usize"); i += 2; }
                "--no-filter-rated"=> { filter_rated = false; i += 1; }
                "--device"         => {
                    use_cuda = raw[i + 1].trim().to_lowercase() == "cuda";
                    i += 2;
                }
                "--save-results"   => { results_path = Some(PathBuf::from(&raw[i + 1])); i += 2; }
                other => {
                    eprintln!("Argomento non riconosciuto: {}", other);
                    std::process::exit(1);
                }
            }
        }

        if num_items == 0 {
            eprintln!("Errore: --num-items è obbligatorio e deve essere > 0");
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

// ---------------------------------------------------------------------------
// Lettura sequenze dal file di testo
// Formato: ogni riga contiene gli ID item separati da spazio (storia di un utente)
// ---------------------------------------------------------------------------

fn read_sequences(path: &PathBuf) -> Vec<Vec<u32>> {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Impossibile aprire {:?}: {}", path, e));
    BufReader::new(file)
        .lines()
        .map(|line| {
            line.unwrap()
                .split_whitespace()
                .filter_map(|s| s.parse::<u32>().ok())
                .collect()
        })
        .collect()
}

fn read_ground_truth(path: &PathBuf) -> Vec<u32> {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Impossibile aprire {:?}: {}", path, e));
    BufReader::new(file)
        .lines()
        .map(|line| line.unwrap().trim().parse::<u32>().unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// Padding / troncamento: stessa logica di dataset.rs
// ---------------------------------------------------------------------------

fn prepare_sequence(seq: &[u32], max_length: usize, pad_val: u32) -> (Vec<u32>, HashSet<u32>) {
    let rated: HashSet<u32> = seq.iter().cloned().collect();
    let mut inp = seq.to_vec();

    if inp.len() > max_length {
        inp = inp[(inp.len() - max_length)..].to_vec();
    } else {
        let diff = max_length - inp.len();
        let mut padded = vec![pad_val; diff];
        padded.extend_from_slice(&inp);
        inp = padded;
    }

    (inp, rated)
}

// ---------------------------------------------------------------------------
// Metriche: NDCG@K e Recall@K
// ---------------------------------------------------------------------------

fn ndcg_at_k(ranked: &[(u32, f32)], relevant: u32, k: usize) -> f64 {
    for (rank, (item, _)) in ranked.iter().take(k).enumerate() {
        if *item == relevant {
            // DCG con rilevanza binaria: 1 / log2(rank + 2)
            return 1.0 / (rank as f64 + 2.0).log2();
        }
    }
    0.0
}

fn recall_at_k(ranked: &[(u32, f32)], relevant: u32, k: usize) -> f64 {
    if ranked.iter().take(k).any(|(item, _)| *item == relevant) {
        1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> candle_core::Result<()> {
    let args = Args::parse();

    // -----------------------------------------------------------------------
    // 1. Seleziona il device
    // -----------------------------------------------------------------------
    let device = if args.use_cuda {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };
    println!("Device: {:?}", device);

    // -----------------------------------------------------------------------
    // 2. Carica i pesi dal file .safetensors
    // -----------------------------------------------------------------------
    println!("Caricamento pesi da {:?} ...", args.model_path);
    let bytes = std::fs::read(&args.model_path)
        .unwrap_or_else(|e| panic!("Impossibile leggere {:?}: {}", args.model_path, e));

    let safetensors = SafeTensors::deserialize(&bytes)
        .expect("File .safetensors non valido o corrotto");

    // Stampa i tensori trovati per debug
    println!("  Tensori trovati nel file:");
    for (name, view) in safetensors.tensors() {
        println!("    {:<55} {:?}", name, view.shape());
    }

    let vb = VarBuilder::from_safetensors(
        vec![safetensors],
        DType::F32,
        &device,
    );

    // -----------------------------------------------------------------------
    // 3. Costruisce il modello con la config
    // -----------------------------------------------------------------------
    let mut config = GsasrecConfig::new("infer", args.num_items);
    config.sequence_length      = args.sequence_length;
    config.embedding_dim        = args.embedding_dim;
    config.num_heads            = args.num_heads;
    config.num_blocks           = args.num_blocks;
    config.dropout_rate         = args.dropout_rate;
    config.reuse_item_embeddings = args.reuse_item_embeddings;
    config.recommendation_limit = args.limit;

    println!("\nConfig modello:");
    println!("  num_items        = {}", config.num_items);
    println!("  sequence_length  = {}", config.sequence_length);
    println!("  embedding_dim    = {}", config.embedding_dim);
    println!("  num_heads        = {}", config.num_heads);
    println!("  num_blocks       = {}", config.num_blocks);
    println!("  reuse_embeddings = {}", config.reuse_item_embeddings);

    let model = GSASRec::new(vb, config.clone())?;
    println!("\n✓ Modello costruito");

    // -----------------------------------------------------------------------
    // 4. Legge le sequenze di input
    // -----------------------------------------------------------------------
    let all_sequences = read_sequences(&args.input_path);
    let n_users = all_sequences.len();
    println!("Utenti da processare: {}", n_users);

    let ground_truth: Option<Vec<u32>> = args.output_path.as_ref().map(read_ground_truth);

    let pad_val = args.num_items + 1;

    // -----------------------------------------------------------------------
    // 5. Inferenza a batch
    // -----------------------------------------------------------------------
    let mut all_results: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n_users);
    let t_start = Instant::now();

    let chunks: Vec<&[Vec<u32>]> = all_sequences.chunks(args.batch_size).collect();
    let n_batches = chunks.len();

    for (batch_idx, chunk) in chunks.iter().enumerate() {
        let b = chunk.len();
        let mut flat_input: Vec<u32> = Vec::with_capacity(b * args.sequence_length);
        let mut rated_batch: Vec<HashSet<u32>> = Vec::with_capacity(b);

        for seq in chunk.iter() {
            let (padded, rated) = prepare_sequence(seq, args.sequence_length, pad_val);
            flat_input.extend_from_slice(&padded);
            rated_batch.push(rated);
        }

        let input_tensor = Tensor::from_vec(
            flat_input,
            (b, args.sequence_length),
            &device,
        )?;

        let rated_opt = if args.filter_rated {
            Some(&rated_batch)
        } else {
            None
        };

        let batch_results = model.get_predictions(&input_tensor, args.limit, rated_opt)?;
        all_results.extend(batch_results);

        if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == n_batches {
            let elapsed = t_start.elapsed().as_secs_f32();
            println!(
                "  Batch {}/{} completato ({:.2}s totali, {:.1} utenti/s)",
                batch_idx + 1,
                n_batches,
                elapsed,
                all_results.len() as f32 / elapsed,
            );
        }
    }

    let total_elapsed = t_start.elapsed().as_secs_f32();
    println!("\n✓ Inferenza completata in {:.2}s ({:.1} utenti/s)",
        total_elapsed,
        n_users as f32 / total_elapsed,
    );

    // -----------------------------------------------------------------------
    // 6. Stampa risultati (prime 5 righe)
    // -----------------------------------------------------------------------
    println!("\nPrime 5 raccomandazioni:");
    for (uid, recs) in all_results.iter().take(5).enumerate() {
        let items: Vec<String> = recs.iter()
            .map(|(id, score)| format!("{}({:.3})", id, score))
            .collect();
        println!("  Utente {:>4}: {}", uid, items.join("  "));
    }

    // -----------------------------------------------------------------------
    // 7. Calcola metriche se è disponibile il ground truth
    // -----------------------------------------------------------------------
    if let Some(ref gt) = ground_truth {
        assert_eq!(
            gt.len(), n_users,
            "Numero di righe in --output ({}) != numero di utenti ({})",
            gt.len(), n_users
        );

        let ks: &[usize] = &[1, 5, 10];
        let mut ndcg   = vec![0.0f64; ks.len()];
        let mut recall = vec![0.0f64; ks.len()];

        for (uid, (recs, &relevant)) in all_results.iter().zip(gt.iter()).enumerate() {
            for (ki, &k) in ks.iter().enumerate() {
                ndcg[ki]   += ndcg_at_k(recs, relevant, k);
                recall[ki] += recall_at_k(recs, relevant, k);
            }
            let _ = uid; // usato implicitamente da enumerate
        }

        println!("\n=== Metriche su {} utenti ===", n_users);
        println!("{:<12} {:>10} {:>10}", "Metrica", "Valore", "");
        println!("{}", "-".repeat(34));
        for (ki, &k) in ks.iter().enumerate() {
            println!(
                "NDCG@{:<7} {:>10.4}",
                k,
                ndcg[ki] / n_users as f64,
            );
            println!(
                "Recall@{:<5} {:>10.4}",
                k,
                recall[ki] / n_users as f64,
            );
        }
    }

    // -----------------------------------------------------------------------
    // 8. Salva i risultati su file (opzionale)
    // Formato: ogni riga = "item_id score  item_id score  ..."
    // -----------------------------------------------------------------------
    if let Some(ref path) = args.results_path {
        let mut f = File::create(path)
            .unwrap_or_else(|e| panic!("Impossibile creare {:?}: {}", path, e));
        for recs in &all_results {
            let line: Vec<String> = recs.iter()
                .map(|(id, score)| format!("{} {:.6}", id, score))
                .collect();
            writeln!(f, "{}", line.join("\t")).unwrap();
        }
        println!("\n✓ Risultati salvati in {:?}", path);
    }

    Ok(())
}