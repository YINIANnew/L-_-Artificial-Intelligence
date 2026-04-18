import math
import random
import numpy as np

class Tensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = np.array([data])
        self.shape = self.data.shape
    
    def __getitem__(self, key):
        result = self.data[key]
        if isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result
    
    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    def transpose(self, dim0, dim1):
        result = self.data.transpose(dim0, dim1)
        return Tensor(result)
    
    def view(self, *shape):
        result = self.data.reshape(shape)
        return Tensor(result)
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    @staticmethod
    def matmul(a, b):
        result = np.matmul(a.data, b.data)
        return Tensor(result)
    
    @staticmethod
    def randn(*shape):
        data = np.random.randn(*shape)
        return Tensor(data)
    
    @staticmethod
    def ones(*shape):
        data = np.ones(shape)
        return Tensor(data)
    
    @staticmethod
    def tril(mat):
        result = np.tril(mat.data)
        return Tensor(result)
    
    def softmax(self, dim=-1):
        result = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        result = result / np.sum(result, axis=dim, keepdims=True)
        return Tensor(result)
    
    def topk(self, k, dim=-1, largest=True, sorted=True):
        values, indices = np.topk(self.data, k, axis=dim, largest=largest, sorted=sorted)
        return Tensor(values), Tensor(indices)
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            result = self.data + other.data
        else:
            result = self.data + other
        return Tensor(result)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            result = self.data * other.data
        else:
            result = self.data * other
        return Tensor(result)
    
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            result = self.data / other.data
        else:
            result = self.data / other
        return Tensor(result)
    
    def masked_fill(self, mask, value):
        result = self.data.copy()
        if isinstance(mask, Tensor):
            result[mask.data] = value
        else:
            result[mask] = value
        return Tensor(result)
    
    def squeeze(self, dim=None):
        result = np.squeeze(self.data, axis=dim)
        return Tensor(result)
    
    def unsqueeze(self, dim):
        result = np.expand_dims(self.data, axis=dim)
        return Tensor(result)
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            result = self.data == other.data
        else:
            result = self.data == other
        return Tensor(result)
    
    def any(self, dim=None):
        return np.any(self.data, axis=dim)
    
    def clamp(self, min=None, max=None):
        result = np.clip(self.data, min, max)
        return Tensor(result)
    
    def dot(self, other):
        if isinstance(other, Tensor):
            return np.dot(self.data.flatten(), other.data.flatten())
        else:
            return np.dot(self.data.flatten(), other)
    
    def sum(self, dim=None, keepdim=False):
        result = np.sum(self.data, axis=dim, keepdims=keepdim)
        if not keepdim and dim is not None:
            return Tensor(result)
        elif isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result
    
    def mean(self, dim=None, keepdim=False):
        result = np.mean(self.data, axis=dim, keepdims=keepdim)
        if not keepdim and dim is not None:
            return Tensor(result)
        elif isinstance(result, np.ndarray):
            return Tensor(result)
        else:
            return result
    
    def expand(self, *shape):
        result = np.broadcast_to(self.data, shape)
        return Tensor(result)
    
    def reshape(self, *shape):
        result = self.data.reshape(shape)
        return Tensor(result)
    
    def transpose(self, *dims):
        result = self.data.transpose(*dims)
        return Tensor(result)
    
    def flatten(self):
        result = self.data.flatten()
        return Tensor(result)
    
    def contiguous(self):
        if not self.data.flags.contiguous:
            return Tensor(self.data.copy())
        return self
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def to(self, dtype):
        result = self.data.astype(dtype)
        return Tensor(result)

class nn:
    class Module:
        def __init__(self):
            pass
        
        def forward(self, *args, **kwargs):
            raise NotImplementedError
        
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
    
    class ReLU:
        def __call__(self, x):
            return F.relu(x)
    
    class GELU:
        def __call__(self, x):
            return F.relu(x)
    
    class Tanh:
        def __call__(self, x):
            def tanh_recursive(data):
                if isinstance(data, list):
                    return [tanh_recursive(item) for item in data]
                return math.tanh(data)
            return Tensor(tanh_recursive(x.data))
    
    class Sigmoid:
        def __call__(self, x):
            def sigmoid_recursive(data):
                if isinstance(data, list):
                    return [sigmoid_recursive(item) for item in data]
                return 1.0 / (1.0 + math.exp(-data))
            return Tensor(sigmoid_recursive(x.data))
    
    class Linear:
        def __init__(self, in_features, out_features, bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            
            self.weight = Tensor.randn(out_features, in_features).data
            if bias:
                self.bias = Tensor.randn(out_features).data
        
        def forward(self, x):
            batch_size = len(x.data)
            seq_len = len(x.data[0]) if len(x.shape) == 3 else 1
            
            if len(x.shape) == 3:
                output = []
                for batch in x.data:
                    batch_output = []
                    for seq in batch:
                        result = [0.0 for _ in range(self.out_features)]
                        for i in range(self.out_features):
                            for j in range(self.in_features):
                                result[i] += self.weight[i][j] * seq[j]
                            if self.bias:
                                result[i] += self.bias[i]
                        batch_output.append(result)
                    output.append(batch_output)
            else:
                output = []
                for batch in x.data:
                    result = [0.0 for _ in range(self.out_features)]
                    for i in range(self.out_features):
                        for j in range(self.in_features):
                            result[i] += self.weight[i][j] * batch[j]
                        if self.bias:
                            result[i] += self.bias[i]
                    output.append(result)
            
            return Tensor(output)
        
        def __call__(self, x):
            return self.forward(x)
    
    class LayerNorm:
        def __init__(self, normalized_shape, eps=1e-5):
            self.normalized_shape = normalized_shape
            self.eps = eps
            self.weight = [1.0 for _ in range(normalized_shape)]
            self.bias = [0.0 for _ in range(normalized_shape)]
        
        def forward(self, x):
            if len(x.shape) == 3:
                output = []
                for batch in x.data:
                    batch_output = []
                    for seq in batch:
                        mean = sum(seq) / len(seq)
                        var = sum((s - mean) ** 2 for s in seq) / len(seq)
                        normalized = [(s - mean) / math.sqrt(var + self.eps) for s in seq]
                        scaled = [n * w + b for n, w, b in zip(normalized, self.weight, self.bias)]
                        batch_output.append(scaled)
                    output.append(batch_output)
            else:
                output = []
                for batch in x.data:
                    mean = sum(batch) / len(batch)
                    var = sum((b - mean) ** 2 for b in batch) / len(batch)
                    normalized = [(b - mean) / math.sqrt(var + self.eps) for b in batch]
                    scaled = [n * w + b for n, w, b in zip(normalized, self.weight, self.bias)]
                    output.append(scaled)
            
            return Tensor(output)
    
    class Dropout:
        def __init__(self, p=0.5):
            self.p = p
        
        def forward(self, x):
            def dropout_recursive(data):
                if isinstance(data, list):
                    return [dropout_recursive(item) for item in data]
                return data if random.random() > self.p else 0.0
            
            return Tensor(dropout_recursive(x.data))
        
        def __call__(self, x):
            return self.forward(x)
    
    class Sequential:
        def __init__(self, *layers):
            self.layers = layers
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
        
        def __call__(self, x):
            return self.forward(x)

class F:
    @staticmethod
    def softmax(x, dim=-1):
        if dim < 0:
            dim = len(x.shape) + dim
        
        def softmax_recursive(data, dim, current_dim=0):
            if not isinstance(data, list):
                return data
            if current_dim == dim:
                exp_vals = [math.exp(d) for d in data]
                sum_exp = sum(exp_vals)
                return [e / sum_exp for e in exp_vals]
            else:
                return [softmax_recursive(item, dim, current_dim + 1) for item in data]
        
        return Tensor(softmax_recursive(x.data, dim))
    
    @staticmethod
    def relu(x):
        def relu_recursive(data):
            if isinstance(data, list):
                return [relu_recursive(item) for item in data]
            return max(0.0, data)
        
        return Tensor(relu_recursive(x.data))
    
    @staticmethod
    def normalize(x, dim=-1):
        def normalize_recursive(data, dim, current_dim=0):
            if current_dim == dim:
                norm = math.sqrt(sum(d ** 2 for d in data))
                return [d / norm for d in data]
            else:
                return [normalize_recursive(item, dim, current_dim + 1) for item in data]
        
        return Tensor(normalize_recursive(x.data, dim))

class LAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = Tensor.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float('inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        output = Tensor.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def get_config(self):
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'bias': self.q_proj.bias is not None
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            dropout=config.get('dropout', 0.1),
            bias=config.get('bias', True)
        )
    
    def save_config(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_config(), f, ensure_ascii=False, indent=2)
    
    def load_config(self, path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return self.from_config(config)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.attention = LAttention(embed_dim, num_heads, dropout, bias)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        output, _ = self.attention(query, key, value, attn_mask, key_padding_mask)
        return output

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        self.attention = LAttention(embed_dim, num_heads, dropout, bias)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        output, _ = self.attention(x, x, x, attn_mask, key_padding_mask)
        return output

class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, top_k=32, dropout=0.1, bias=True):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = Tensor.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, -float('inf'))
        
        top_k = min(self.top_k, seq_len_k)
        top_attn_scores = []
        top_indices = []
        
        for batch in attn_scores.data:
            batch_top_scores = []
            batch_top_indices = []
            for head in batch:
                head_top_scores = []
                head_top_indices = []
                for seq in head:
                    sorted_indices = sorted(range(len(seq)), key=lambda i: seq[i], reverse=True)[:top_k]
                    head_top_indices.append(sorted_indices)
                    head_top_scores.append([seq[i] for i in sorted_indices])
                batch_top_scores.append(head_top_scores)
                batch_top_indices.append(head_top_indices)
            top_attn_scores.append(batch_top_scores)
            top_indices.append(batch_top_indices)
        
        top_attn_scores = Tensor(top_attn_scores)
        
        attn_weights = F.softmax(top_attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        output = []
        for batch_idx, batch in enumerate(v.data):
            batch_output = []
            for head_idx, head in enumerate(batch):
                head_output = []
                for seq_idx, seq in enumerate(head):
                    indices = top_indices[batch_idx][head_idx][seq_idx]
                    selected_v = [head[i] for i in indices]
                    weighted_sum = [0.0 for _ in range(self.head_dim)]
                    for i, weight in enumerate(attn_weights.data[batch_idx][head_idx][seq_idx]):
                        for j, val in enumerate(selected_v[i]):
                            weighted_sum[j] += weight * val
                    head_output.append(weighted_sum)
                batch_output.append(head_output)
            output.append(batch_output)
        
        output = Tensor(output).transpose(1, 2).view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

class LinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = F.relu(q)
        k = F.relu(k)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            k = k.masked_fill(key_padding_mask, 0)
            v = v.masked_fill(key_padding_mask, 0)
        
        kv = []
        for batch in range(batch_size):
            batch_kv = []
            for head in range(self.num_heads):
                head_k = k.data[batch][head]
                head_v = v.data[batch][head]
                head_k_t = list(map(list, zip(*head_k)))
                head_kv = [[0.0 for _ in range(self.head_dim)] for _ in range(self.head_dim)]
                for i in range(self.head_dim):
                    for j in range(self.head_dim):
                        for k_idx in range(seq_len_k):
                            head_kv[i][j] += head_k_t[i][k_idx] * head_v[k_idx][j]
                batch_kv.append(head_kv)
            kv.append(batch_kv)
        kv = Tensor(kv)
        
        qkv = []
        for batch in range(batch_size):
            batch_qkv = []
            for head in range(self.num_heads):
                head_q = q.data[batch][head]
                head_kv = kv.data[batch][head]
                head_qkv = []
                for seq in head_q:
                    seq_qkv = [0.0 for _ in range(self.head_dim)]
                    for i in range(self.head_dim):
                        for j in range(self.head_dim):
                            seq_qkv[i] += seq[j] * head_kv[j][i]
                    head_qkv.append(seq_qkv)
                batch_qkv.append(head_qkv)
            qkv.append(batch_qkv)
        qkv = Tensor(qkv)
        
        k_sum = []
        for batch in range(batch_size):
            batch_k_sum = []
            for head in range(self.num_heads):
                head_k = k.data[batch][head]
                head_k_sum = [0.0 for _ in range(self.head_dim)]
                for seq in head_k:
                    for i in range(self.head_dim):
                        head_k_sum[i] += seq[i]
                batch_k_sum.append([head_k_sum])
            k_sum.append(batch_k_sum)
        k_sum = Tensor(k_sum)
        
        qk_sum = []
        for batch in range(batch_size):
            batch_qk_sum = []
            for head in range(self.num_heads):
                head_q = q.data[batch][head]
                head_k_sum = k_sum.data[batch][head]
                head_qk_sum = []
                for seq in head_q:
                    seq_qk_sum = 0.0
                    for i in range(self.head_dim):
                        seq_qk_sum += seq[i] * head_k_sum[0][i]
                    head_qk_sum.append([seq_qk_sum])
                batch_qk_sum.append(head_qk_sum)
            qk_sum.append(batch_qk_sum)
        qk_sum = Tensor(qk_sum)
        qk_sum = qk_sum + 1e-8
        
        output = qkv / qk_sum
        
        output = output.transpose(1, 2).view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)
        
        return output, None

class AttentionSelector:
    def __init__(self):
        pass
    
    @staticmethod
    def select_attention(embed_dim, num_heads, seq_len, hardware='auto'):
        if seq_len > 1000:
            print(f"Selecting LinearAttention for long sequence (seq_len={seq_len})")
            return LinearAttention(embed_dim, num_heads)
        elif seq_len > 500:
            print(f"Selecting SparseAttention for medium sequence (seq_len={seq_len})")
            return SparseAttention(embed_dim, num_heads)
        else:
            print(f"Selecting LAttention for short sequence (seq_len={seq_len})")
            return LAttention(embed_dim, num_heads)