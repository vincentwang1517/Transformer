import torch
import torch.nn as nn
import math

'''
Input Embeddings
'''
class InputEmbeddings(nn.Module):
    '''
    vocab_size: The size of your dictionary
    d_model: The dimension of the embedded output
    '''
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # variable
        self.voacb_size = vocab_size
        self.d_model = d_model
        # model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

'''
Positional Encoding
'''
class PositionalEncoding(nn.Module):
    '''
    seq_len: The maximum length of the sentence
    '''
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # In positional encoding, we add the matrix pos (seq_len, d_model) to the embedded input
        pe = torch.zeros(seq_len, d_model) # (seq_len, d_model)
        # pe formula
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(-1 * torch.arange(0, d_model, 2, dtype=torch.float) * math.log(10000.0) / d_model) # (d_model / 2, )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # The actual shape is (batch, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # not a trainable nn.Parameter -> need to be explicitly registered
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

'''
Layer Normalization
'''
class LayerNormalization(nn.Module):
    
    def __init__(self, eps :float = 10**-6):
        super().__init__()
        self.eps = eps
        # introduce fluctuation
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        ''' x: input #(batch, seq_len. d_model)'''
        mean = torch.mean(x, dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = torch.std(x, dim=-1, keepdim=True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
        
'''
Feed Forward
'''
class FeedFowardBlock(nn.Module):
    '''
    d_ff: The dimension of the intermidiate layer
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        #model
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 & b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

'''
Multi-Head Attention
'''
class MultiHeadAttentionBlock(nn.Module):
    '''
    h: The head number
    '''
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # Leanable Matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        '''
        [Input]
        query, key, value: (batch, h, seq_len, d_k)
        [Output]
        value with attention (batch, h, seq_len, d_k)
        attention scores
        '''
        d_k = query.shape[-1]
        
        # Attention scores
        # Q x K.transpose() :: For row 0, use query 0 and it's match score to all keys_j
        attention_scores = query @ key.transpose(-1, -2) # (batch, h, seq_len, seq_len) :: Q x K.transpose()
        attention_scores = attention_scores / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.masked_fill(attention_scores, mask==0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores) # (batch, h, seq_len, seq_len)
            
        return attention_scores @ value, attention_scores
        
    def forward(self, q, k, v, mask):
        # Q' = q x w_q
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # multi-head... desired shape (batch, head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # (batch, h, seq_len, d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(x) # (batch, seq_len, d_model)

'''
Residual Connection
'''
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))
        
        