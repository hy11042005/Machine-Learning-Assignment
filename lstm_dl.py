import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# =====================
# 1. LOAD DATASET
# =====================
df = pd.read_csv("IMDb Dataset.csv")

df = df.rename(columns={"review": "text", "sentiment": "label"})
df["label"] = df["label"].map({"positive": 1, "negative": 0})

# =====================
# 2. CLEAN TEXT
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

texts = df["text"].values
labels = df["label"].values

# =====================
# 3. TOKENIZATION
# =====================
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 200

word2idx = {"<PAD>": 0, "<UNK>": 1}
idx = 2

for text in texts:
    for word in text.split():
        if word not in word2idx:
            if idx < MAX_VOCAB_SIZE:
                word2idx[word] = idx
                idx += 1

def encode_text(text):
    tokens = []
    for word in text.split():
        tokens.append(word2idx.get(word, 1))
    return tokens

encoded_texts = [encode_text(text) for text in texts]

def pad_sequence(seq, max_len):
    if len(seq) < max_len:
        return seq + [0] * (max_len - len(seq))
    else:
        return seq[:max_len]

padded_texts = np.array([pad_sequence(seq, MAX_SEQ_LEN) for seq in encoded_texts])

# =====================
# 4. TRAIN TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    padded_texts,
    labels,
    test_size=0.2,
    random_state=42
)

# =====================
# 5. DATASET + DATALOADER
# =====================
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# =====================
# 6. LSTM MODEL
# =====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn[-1])
        out = self.fc(hn)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    vocab_size=MAX_VOCAB_SIZE,
    embed_dim=128,
    hidden_dim=128,
    num_classes=2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 7. TRAINING
# =====================
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# =====================
# 8. EVALUATION
# =====================
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())

print("\n===== LSTM RESULT =====")
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
