# aipredictor
Simple python program to demo transformer AI learning and prediction.

## Project Overview

This project implements a **general-purpose sequence prediction system** using a **Transformer-based model**. Originally developed for predicting the next notes in musical melodies (MIDI), it has been designed to handle multiple types of sequential data, including:

- **MIDI melodies** (integers representing notes)
- **Stock prices** (floating-point values)
- **DNA sequences** (categorical symbols)

The project is modular, allowing domain-specific preprocessing and data input while reusing the same core transformer logic.

---

## Goals

1. **Train a Transformer model** on sequential data to predict the next item in a sequence.
2. **Handle multiple sequence types**:
    - `INTEGER` → discrete sequences like MIDI notes
    - `FLOAT` → continuous sequences like stock prices
    - `TUPLE` → categorical sequences like DNA symbols
3. **Allow user-defined parameters** for sequence length, number of training epochs, and prediction steps.
4. **Be extensible** to different domains and support future UI integration (e.g., via Tkinter).

---

## Project Structure
```
project/
│
├─ xformer.py # Core Transformer module (training, prediction, preprocessing)
├─ predict_melodies_main.py # Example main script for music melodies
├─ predict_stock_main.py # Example main script for stock data (future)
├─ predict_dna_main.py # Example main script for DNA sequences (future)
├─ requirements.txt # Project dependencies
└─ utils/ # Optional helper functions for file reading, MIDI parsing, etc.
```

---

## Key Components

### 1. `xformer.py`

- Implements `SequenceTransformer`, a mini Transformer model:
    - Embedding layer
    - Positional encoding
    - Transformer layers
    - Fully connected output
- Handles:
    - Creating sequences (`create_sequences`)
    - Preprocessing:
        - Mapping integers or categorical symbols to indices
        - Normalizing floats
    - Training with cross-entropy loss
    - Generating predicted sequences
- Accepts configurable parameters:
    - `input_data_type` (INTEGER, FLOAT, TUPLE)
    - `seq_length` (length of input sequences)
    - `epochs` (training iterations)
    - `generate_steps` (length of generated output)

### 2. Main Scripts

- **`predict_melodies_main.py`**
    - Reads melodies (simulated or MIDI)
    - Converts to integers
    - Calls `train_and_predict()` from `xformer.py`
    - Prints generated continuation

- **`predict_stock_main.py`**
    - (Future) Reads stock prices CSV
    - Normalizes float values
    - Calls `train_and_predict()` from `xformer.py`
    
- **`predict_dna_main.py`**
    - (Future) Reads DNA sequences
    - Maps symbols to integers
    - Calls `train_and_predict()` from `xformer.py`

---

## Workflow

1. **Preprocess sequences**:
    - Map to integers (MIDI, DNA) or normalize floats (stock data)
    - Detect unique values or min/max ranges
2. **Create input-target pairs** using sliding window (`seq_length`)
3. **Train Transformer**:
    - Predict the next item
    - Calculate loss (cross-entropy)
    - Update model weights using backpropagation (Adam optimizer)
4. **Generate predictions**:
    - Seed with initial sequence
    - Iteratively predict next item
    - Convert back to original scale or mapping if needed

---

## Global Configuration Parameters

- `INPUT_DATA_TYPE` = "INTEGER" | "FLOAT" | "TUPLE"
- `SEQ_LENGTH` = number of items in input sequences (default: 4)
- `EPOCHS` = number of training iterations (default: 200)
- `GENERATE_STEPS` = number of items to predict after training (default: 20)

These can be modified in main scripts or passed in dynamically for future UI integration.


## Dependencies

torch>=2.0.0
numpy>=1.25.0
# optional for MIDI: mido>=1.3.0, pretty_midi>=0.3.10

## Install
1_create_env.bat  (only run this once)
2_activate_env.bat
3_install_dependencies.bat

## Extensibility
- New data domains can be added by writing a small preprocessing main script (e.g., predict_stock_main.py) while reusing the core xformer.py.
- Supports automatic preprocessing:
    - Integers: map to indices
    - Floats: normalize to 0–1
    - Categorical: map symbols to indices
- Allows future UI integration for parameter input and sequence generation.

## Educational Purpose
- Demonstrates Transformer sequence learning in a concrete, approachable way.
- Shows how patterns in sequences (music, DNA, stock trends) can be learned and predicted.
- Helps students understand general-purpose AI sequence prediction, not limited to text.

## UML
```
          +----------------------+
          | predict_*_main.py    |
          | (melodies, stock,    |
          |  DNA, etc.)          |
          +----------+-----------+
                     |
                     | calls train_and_predict()
                     v
          +----------------------+
          |    xformer.py        |
          | -------------------  |
          | SequenceTransformer  |
          |  - embedding         |
          |  - positional encode |
          |  - transformer layers|
          |  - fc output         |
          +----------+-----------+
                     |
    +----------------+----------------+
    |                                 |
+---v---+                       +-----v------+
| model |                       | optimizer  |
| (nn.Module)                   | (Adam)     |
+-------+                       +------------+
    |
    v
+---v---+
| criterion |
| (CrossEntropyLoss) |
+-----------+

## Sequence Flow Diagram (ASCII)

```text
Input Sequence (length = seq_length)
      |
      v
+----------------+
|  Embedding     |  <-- Maps each value/note to a vector
+----------------+
      |
      v
+----------------+
| Positional     |  <-- Adds position info so model knows order
| Encoding       |
+----------------+
      |
      v
+----------------+
| Transformer    |  <-- Multi-head self-attention + feedforward layers
| Layers         |
+----------------+
      |
      v
+----------------+
| Fully Connected|  <-- Projects to output space (vocab size / value range)
| Output Layer   |
+----------------+
      |
      v
Predicted Next Item(s)
      |
      v
(Optional) Loop back to include prediction in next input
for sequence generation of length generate_steps
```

- predict_*_main.py scripts handle data preprocessing and call the core transformer module.
- SequenceTransformer contains the embedding, positional encoding, transformer layers, and output layer.
- The optimizer and criterion are used during training to update model weights.