from xformer import train_and_predict
import random

# User can pass parameters, defaults used otherwise
SEQ_LENGTH = 4
EPOCHS = 200
GENERATE_STEPS = 20
INPUT_DATA_TYPE = "INTEGER"

# Example: generate 30 random melodies of variable lengths
NUM_MELODIES = 30
MIN_MELODY_LENGTH = 5
MAX_MELODY_LENGTH = 50
melodies = []

for _ in range(NUM_MELODIES):
    length = random.randint(MIN_MELODY_LENGTH, MAX_MELODY_LENGTH)
    base = [60,62,64,65,67,65,64,62,60,62]
    melody = [random.choice(base)+random.randint(-2,2) for _ in range(length)]
    melodies.extend(melody)  # flatten all melodies for simplicity

# Train and predict
generated = train_and_predict(
    sequence_data=melodies,
    input_data_type=INPUT_DATA_TYPE,
    seq_length=SEQ_LENGTH,
    epochs=EPOCHS,
    generate_steps=GENERATE_STEPS
)

print("Generated melody continuation:")
print(generated)