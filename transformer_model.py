import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_k):
        super(Attention, self).__init__()
        self.d_k = d_k  # dimension of keys

    def forward(self, Q, K, V, mask=None):
        # Q: [batch, heads, seq_len, d_k]
        # K: [batch, heads, seq_len, d_k]
        # V: [batch, heads, seq_len, d_v]

        # Step 1: Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Step 2: Apply mask (optional, for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Step 3: Softmax normalization (optinal, for numerical stability)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]

        # Step 4: Weighted sum of values
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, d_v]
        
        return output, attn_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.d_k = model_dim // num_heads
        self.num_heads = num_heads
        
        self.linear_Q = nn.Linear(model_dim, model_dim)
        self.linear_K = nn.Linear(model_dim, model_dim)
        self.linear_V = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)
        
        self.attention = Attention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Step 1: Linear projections
        Q = self.linear_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        K = self.linear_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        V = self.linear_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, seq_len, d_v]
        
        # Step 2: Apply attention
        # attnweights is for  later direct concatenation (in different archtecture)
        # so in this case we are not using it
        attn_output, attnweights  = self.attention(Q, K, V, mask)  # [batch, heads, seq_len, d_v]
        
        # Step 3: Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)  # [batch, seq_len, model_dim]
        output = self.linear_out(attn_output)  # [batch, seq_len, model_dim]
        
        return output
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim=1):
        super(TransformerModel, self).__init__()
        # can be replaced with pre-trained embedding layer (rememeber to adjust input_dim accordingly and language)
        # example: nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        # here we just use a simple embedding layer
        self.embedding = nn.Embedding(input_dim, model_dim)

        self.layers = nn.ModuleList([MultiHeadAttention(model_dim, num_heads) for _ in range(num_layers)])
        # task specific output layer
        # in this case we are doing probaility distribution over output classes
        # or regression task with single output
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src, mask=None):
        embedded = self.embedding(src)  # [batch, seq_len, model_dim]
        
        x = embedded
        for layer in self.layers:
            x = layer(x, x, x, mask)  # Self-attention
        
        output = self.fc_out(x)  # [batch, seq_len, output_dim]
        return output
    
# Example usage:
model = TransformerModel(input_dim=1000, model_dim= 16, num_heads= 8, num_layers=6, output_dim=1)
src = torch.randint(0, 1000, (32, 20))  # [batch_size, seq_len]
print(src)
out = model(src)  # [batch_size, seq_len, output_dim] 
print("handling")
print(out)
