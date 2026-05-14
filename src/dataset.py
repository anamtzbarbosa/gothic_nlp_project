import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class GothicDataset(Dataset):
    def __init__(self, token_list, seq_length):
        self.tokens = token_list
        self.seq_length = seq_length # window

    def __len__(self):
        # num windows we can make
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        # x: actual sequence
        # y: sequence shifted one to right
        x = torch.tensor(self.tokens[idx : idx + self.seq_length])
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_length + 1])
        return x, y


def split_tokens_by_chunks(tokens, chunk_size=5000, train_ratio=0.8, val_ratio=0.1, seed=42):
    generator = torch.Generator().manual_seed(seed)

    chunks = [
        tokens[i:i + chunk_size]
        for i in range(0, len(tokens), chunk_size)
        if len(tokens[i:i + chunk_size]) >= chunk_size
    ]

    indices = torch.randperm(len(chunks), generator=generator).tolist()

    train_end = int(len(indices) * train_ratio)
    val_end = int(len(indices) * (train_ratio + val_ratio))

    train_tokens = []
    val_tokens = []
    test_tokens = []

    for i in indices[:train_end]:
        train_tokens.extend(chunks[i])

    for i in indices[train_end:val_end]:
        val_tokens.extend(chunks[i])

    for i in indices[val_end:]:
        test_tokens.extend(chunks[i])

    return train_tokens, val_tokens, test_tokens

def get_dataloaders(path, seq_length, batch_size = 64, train_split=0.8):
    with open(path, 'rb') as f:
        tokens = pickle.load(f)

    n = len(tokens)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_tokens, val_tokens, test_tokens = split_tokens_by_chunks(tokens)

    train_ds = GothicDataset(train_tokens, seq_length)
    val_ds = GothicDataset(val_tokens, seq_length)
    test_ds = GothicDataset(test_tokens, seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    path = 'data/corpus_tokenized.pkl'
    try:
        train_loader, val_loader, test_loader = get_dataloaders(path, seq_length=100, batch_size=32)
        print(f"DataLoaders ready")

        print(f"Batches en Train: {len(train_loader)}")
        print(f"Batches en Val: {len(val_loader)}")
        print(f"Batches en Test: {len(test_loader)}")

    except:
        print("Error")
