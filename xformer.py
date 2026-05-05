import torch
import torch.nn as nn
import torch.optim as optim

class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size, seq_length, embed_size=32, num_heads=2, num_layers=2):
        super().__init__()
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Embedding(seq_length, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, batch_first=True)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src):
        positions = torch.arange(self.seq_length, device=src.device).unsqueeze(0)
        x = self.embedding(src) + self.pos_encoder(positions)
        out = self.transformer(x, x)
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return X, y


def preprocess_sequence(data, input_data_type):
    if input_data_type == "INTEGER":
        unique_vals = sorted(set(data))
        mapping = {val: i for i, val in enumerate(unique_vals)}
        processed = [mapping[val] for val in data]
        return processed, mapping, None
    elif input_data_type == "FLOAT":
        min_val, max_val = min(data), max(data)
        processed = [(val - min_val)/(max_val - min_val) for val in data]
        return processed, None, (min_val, max_val)
    elif input_data_type == "TUPLE":
        unique_vals = sorted(set(data))
        mapping = {val: i for i, val in enumerate(unique_vals)}
        processed = [mapping[val] for val in data]
        return processed, mapping, None
    else:
        raise ValueError("Unsupported input_data_type")


def train_and_predict(
    sequence_data,
    input_data_type="INTEGER",
    seq_length=4,
    epochs=200,
    generate_steps=20
):
    # Preprocess
    processed, mapping, minmax = preprocess_sequence(sequence_data, input_data_type)
    X, y = create_sequences(processed, seq_length)
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    vocab_size = max(processed)+1 if input_data_type != "FLOAT" else len(set(processed))

    # Model
    model = SequenceTransformer(vocab_size, seq_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Prediction: seed with first seq_length items
    seed = processed[:seq_length]
    generated = seed.copy()
    for _ in range(generate_steps):
        seq_input = torch.tensor([generated[-seq_length:]], dtype=torch.long)
        with torch.no_grad():
            pred = model(seq_input)
            next_item = torch.argmax(pred, dim=1).item()
        generated.append(next_item)

    # Convert back to original values if mapping or normalization exists
    if mapping:
        inv_map = {v:k for k,v in mapping.items()}
        generated = [inv_map[i] for i in generated]
    elif minmax:
        min_val, max_val = minmax
        generated = [i*(max_val-min_val)+min_val for i in generated]

    return generated