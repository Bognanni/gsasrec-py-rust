import optuna
import subprocess
import re
import sys

# script for config_optuna.py, file used to specify the hyperparameters of the model each time
TEMPLATE_CONFIG = """from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    train_batch_size={train_batch_size}, 
    sequence_length={sequence_length}, 
    embedding_dim={embedding_dim},
    num_heads={num_heads},
    max_batches_per_epoch=100,
    max_epochs=100,
    num_blocks={num_blocks},
    dropout_rate={dropout_rate},
    negs_per_pos=256,
    gbce_t={gbce_t},
)
"""


def objective(trial):
    # hyperparameters and the list of possible values for each one
    gbce_t = trial.suggest_categorical('gbce_t', [0.0, 0.5, 0.75, 1.0])
    num_blocks = trial.suggest_int('num_blocks', 1, 3)
    embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256])
    num_heads = trial.suggest_categorical('num_heads', [2, 4])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    train_batch_size = trial.suggest_categorical('train_batch_size', [64, 128])
    sequence_length = trial.suggest_categorical('sequence_length', [50, 100, 200])

    # fill the template with the values
    generated_config = TEMPLATE_CONFIG.format(
        sequence_length=sequence_length,
        train_batch_size=train_batch_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
        gbce_t=gbce_t
    )

    # 4. Creiamo un file univoco per ogni tentativo (così puoi ispezionarli)
    name_file_config = "config_optuna.py"

    with open(name_file_config, "w") as file:
        file.write(generated_config)

    print(f"\n[Trial {trial.number}] Created {name_file_config} with: emb={embedding_dim}, batch={train_batch_size}, heads={num_heads}, drop={dropout_rate:.2f}")

    # run train_gsasrec.py from command line specifying the config file
    command = [sys.executable, 'train_gsasrec.py', '--config', name_file_config]

    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    output_text = result.stdout
    # save also the errors
    error_text = result.stderr

    # search for the result
    match = re.search(r'FINAL_SCORE_FOR_OPTUNA: ([0-9]+\.[0-9]+)', output_text)

    if match:
        score = float(match.group(1))
        print(f"[Trial {trial.number}] Score: {score}")
    else:
        print(f"[Trial {trial.number}] Score not found! Consider 0.")
        score = 0.0
        # debug section
        print("\n" + "!" * 40)
        print("Error of the model!")
        print("Output of train_gsasrec.py:")
        print("-" * 40)
        print(error_text)
        print("!" * 40 + "\n")

    return score


# start the Optuna Studio
if __name__ == "__main__":
    print("Starting the hyperparameters optimization.")
    studio = optuna.create_study(direction='maximize')
    studio.optimize(objective, n_trials=5)

    print("\n" + "=" * 40)
    print(f"Fine tuning completed! Best score: {studio.best_value}")
    print("Best params: ", studio.best_params)
    print("=" * 40)