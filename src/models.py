import torch
import torch.nn as nn, torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(VanillaRNN, self).__init__() 
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hidden_state=None):
        embedded_text = self.embed(x)
        seq_output, final_hidden_state = self.rnn(embedded_text, hidden_state)
        logits = self.fc(seq_output)
        return logits, final_hidden_state
    
class DeepLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(DeepLSTM, self).__init__()

        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden_state=None):
        embedded_text = self.embed(x)
        seq_output, final_hidden_state = self.lstm(embedded_text, hidden_state)
        normalized_output = self.norm(seq_output)
        dropped_out_output = self.dropout(normalized_output)
        logits = self.fc(dropped_out_output)
        return logits, final_hidden_state

class CustomCrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CustomCrossAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim,hidden_dim)
        self.W_k = nn.Linear(hidden_dim,hidden_dim)
        self.W_v = nn.Linear(hidden_dim,hidden_dim)
        self.scale = torch.math.sqrt(hidden_dim)

    def forward(self,lstm_output, window_size_k):
        batch_size, seq_len, hidden_dim = lstm_output.shape
        Q = self.W_q(lstm_output)
        K = self.W_k(lstm_output)
        V = self.W_v(lstm_output)
        # to check how much word 1 likes word 2 we do W1.Q * W2.K
        # to get the netire sentence at once do Q*K.T
        K_transposed = K.transpose(1,2) #transpose last 2 dim
        attention_scores = torch.bmm(Q,K_transposed)
        attention_scores = attention_scores/self.scale
        # causal and window masking
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=lstm_output.device)
        mask = torch.tril(mask, diagonal=0) # prevents looking into the future
        if window_size_k > 0:
            mask = torch.triu(mask, diagonal=-window_size_k) # cut off anything older than k steps
        # apply mask, set illegal connections to -infinity so softmax makes them 0
        scores = attention_scores.masked_fill(~mask, float('inf'))
        attention_weights = F.softmax(scores,dim=-1)
        context = torch.bmm(attention_weights, V)
        return context, attention_weights
    


class CrossAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3, window_size_k=5):
        super(CrossAttentionLSTM, self).__init__()
        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = CustomCrossAttention(hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)         
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        attention_context, attention_weights = self.attention(lstm_out, self.window_size_k)
        combined = torch.cat((lstm_out, attention_context), dim=-1)
        normalized_output = self.layer_norm(combined)
        dropped_out_output = self.dropout(normalized_output)
        logits = self.fc(dropped_out_output)
        return logits, hidden, attention_weights


if __name__ == "__main__":
    print("Testing with dummy data...")
    
    # Fake hyperparameters
    VOCAB_SIZE = 5000
    BATCH_SIZE = 16
    SEQ_LEN = 30
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    WINDOW_K = 5
    dummy_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    model = CrossAttentionLSTM(
        vocab_size=VOCAB_SIZE, 
        embed_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        window_size_k=WINDOW_K
    )
    
    logits, hidden, attn_weights = model(dummy_x)
    
    print(f"Input Shape: {dummy_x.shape}")
    print(f"Logits Shape: {logits.shape} -> Expected: (16, 30, 5000)")
    print(f"Attention Weights Shape: {attn_weights.shape} -> Expected: (16, 30, 30)")
    
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), "Logits shape mismatch!"
    
    dummy_loss = logits.sum()
    dummy_loss.backward()
    
    gradient_check = model.lstm.weight_ih_l0.grad is not None
    print(f"\nGradient Flow Check Passed: {gradient_check}")
    
    if gradient_check:
        print("Successful check")
