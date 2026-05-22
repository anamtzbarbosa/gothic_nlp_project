import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.0):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hidden_state=None):
        embedded_text = self.embed(x)
        seq_output, final_hidden_state = self.rnn(embedded_text, hidden_state)
        logits = self.fc(self.dropout(seq_output))
        return logits, final_hidden_state
    
class DeepLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(DeepLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        embedded_text = self.embed(x)
        seq_output, final_hidden_state = self.lstm(embedded_text, hidden)
        normalized_output = self.norm(seq_output)
        dropped_out_output = self.dropout(normalized_output)
        logits = self.fc(dropped_out_output)
        return logits, final_hidden_state

class WindowedSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(WindowedSelfAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        
    def forward(self, lstm_output, window_size_k):
        seq_len = lstm_output.shape[1]
        Q = self.W_q(lstm_output)
        K = self.W_k(lstm_output)
        V = self.W_v(lstm_output)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=lstm_output.device), diagonal=-1)
        if window_size_k > 0:
            mask = mask & torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=lstm_output.device), diagonal=-window_size_k)
            
        scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        return torch.bmm(attention_weights, V), attention_weights

class SelfAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3, window_size_k=5):
        super(SelfAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = WindowedSelfAttention(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        attention_context, attention_weights = self.attention(lstm_out, self.window_size_k)
        
        combined = torch.cat((lstm_out, attention_context), dim=-1)
        normalized_output = self.layer_norm(combined)
        dropped_out_output = self.dropout(normalized_output)
        logits = self.fc(dropped_out_output)
        return logits, hidden, attention_weights

class CachedCrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CachedCrossAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5 

    def forward(self, lstm_output, cached_memory, window_size_k):
        seq_len, hidden_dim = lstm_output.shape[1], lstm_output.shape[2]
        full_context = torch.cat((cached_memory, lstm_output), dim=1) if cached_memory is not None else lstm_output
        memory_len = full_context.shape[1] - seq_len
        
        Q, K, V = self.W_q(lstm_output), self.W_k(full_context), self.W_v(full_context)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=lstm_output.device), diagonal=-1)
        if memory_len > 0:
            mask = torch.cat((torch.ones((seq_len, memory_len), dtype=torch.bool, device=lstm_output.device), mask), dim=1)
        
        if window_size_k > 0:
            window_mask = torch.triu(torch.ones((seq_len, full_context.shape[1]), dtype=torch.bool, device=lstm_output.device), diagonal=-window_size_k + memory_len)
            mask = mask & window_mask
            
        scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        
        return torch.bmm(attention_weights, V), attention_weights, full_context[:, -window_size_k:, :].detach()

class CrossAttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3, window_size_k=5):
        super(CrossAttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.window_size_k = window_size_k
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = CachedCrossAttention(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, x, hidden=None, memory=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        attention_context, attention_weights, new_memory = self.attention(lstm_out, memory, self.window_size_k)
        
        combined = torch.cat((lstm_out, attention_context), dim=-1)
        normalized_output = self.layer_norm(combined)
        dropped_out_output = self.dropout(normalized_output)
        logits = self.fc(dropped_out_output)
        return logits, hidden, attention_weights, new_memory