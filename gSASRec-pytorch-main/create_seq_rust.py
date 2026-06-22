import json
import random


NUM_USERS = 16
MIN_SEQ_LEN = 40
MAX_SEQ_LEN = 40
MAX_ITEM_ID = 3416

sequences = []

for _ in range(NUM_USERS):
    seq_len = random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)

    user_history = [random.randint(1, MAX_ITEM_ID - 1) for _ in range(seq_len)]
    sequences.append(user_history)

payload = {
    "sequences": sequences
}


with open("payload.json", "w") as f:
    json.dump(payload, f)

print(f"File payload.json created with success! (Users: {NUM_USERS})")