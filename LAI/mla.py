import math
import json

import sys
sys.path.append(sys.path[0])

from L_attention import Tensor, nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = q.matmul(k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = attn_weights.matmul(v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights

    def get_config(self):
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout.p
        }


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        batch_size, query_len, _ = query.shape
        kv_len = key_value.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = q.matmul(k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = attn_weights.matmul(v)

        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights

    def get_config(self):
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout.p
        }


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.self_attention = SelfAttention(embed_dim, num_heads, dropout)
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)

    def forward(self, query, key_value=None, attn_mask=None, key_padding_mask=None, use_cross_attention=False):
        if use_cross_attention and key_value is not None:
            return self.cross_attention(query, key_value, attn_mask, key_padding_mask)
        else:
            return self.self_attention(query, attn_mask, key_padding_mask)

    def get_config(self):
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'self_attention_config': self.self_attention.get_config(),
            'cross_attention_config': self.cross_attention.get_config()
        }


class AttentionMasking:
    @staticmethod
    def create_causal_mask(seq_len, device=None):
        mask = Tensor([[1.0 if i >= j else 0.0 for j in range(seq_len)] for i in range(seq_len)])
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    @staticmethod
    def create_causal_mask_batch(batch_size, seq_len, device=None):
        mask = Tensor([[[1.0 if i >= j else 0.0 for j in range(seq_len)] for i in range(seq_len)] for _ in range(batch_size)])
        return mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    @staticmethod
    def create_padding_mask(padding_token_id, batch_size, seq_len, device=None):
        mask = Tensor([[1.0 for _ in range(seq_len)] for _ in range(batch_size)])
        return mask

    @staticmethod
    def create_key_padding_mask(key_padding_tokens, padding_token_id, device=None):
        mask = (key_padding_tokens == padding_token_id)
        return mask

    @staticmethod
    def combine_masks(*masks):
        combined_mask = None
        for mask in masks:
            if mask is None:
                continue
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask + mask
        return combined_mask


class MLALayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ffn_dim=2048, dropout=0.1, use_cross_attention=False):
        super(MLALayer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_cross_attention = use_cross_attention

        self.self_attention = SelfAttention(embed_dim, num_heads, dropout)
        self.self_attention_norm = nn.LayerNorm(embed_dim)

        if use_cross_attention:
            self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
            self.cross_attention_norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None,
                self_key_padding_mask=None, cross_key_padding_mask=None):
        residual = x
        x = self.self_attention_norm(x)
        x, _ = self.self_attention(x, self_attn_mask, self_key_padding_mask)
        x = residual + x

        if self.use_cross_attention and encoder_output is not None:
            residual = x
            x = self.cross_attention_norm(x)
            x, _ = self.cross_attention(x, encoder_output, cross_attn_mask, cross_key_padding_mask)
            x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'use_cross_attention': self.use_cross_attention,
            'self_attention_config': self.self_attention.get_config()
        }
        if self.use_cross_attention:
            config['cross_attention_config'] = self.cross_attention.get_config()
        return config


class MLA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ffn_dim=2048, dropout=0.1,
                 architecture='encoder-decoder', vocab_size=None):
        super(MLA, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.architecture = architecture

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        if architecture == 'encoder-decoder':
            self.encoder_layers = []
            for _ in range(num_layers):
                self.encoder_layers.append(MLALayer(embed_dim, num_heads, ffn_dim, dropout, use_cross_attention=False))
            self.decoder_layers = []
            for _ in range(num_layers):
                self.decoder_layers.append(MLALayer(embed_dim, num_heads, ffn_dim, dropout, use_cross_attention=True))
        elif architecture == 'encoder':
            self.layers = []
            for _ in range(num_layers):
                self.layers.append(MLALayer(embed_dim, num_heads, ffn_dim, dropout, use_cross_attention=False))
        elif architecture == 'decoder':
            self.layers = []
            for _ in range(num_layers):
                self.layers.append(MLALayer(embed_dim, num_heads, ffn_dim, dropout, use_cross_attention=False))

        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_output=None, self_attn_mask=None, cross_attn_mask=None,
                self_key_padding_mask=None, cross_key_padding_mask=None):
        if self.architecture == 'encoder-decoder':
            for layer in self.encoder_layers:
                x = layer(x, self_attn_mask=self_attn_mask, self_key_padding_mask=self_key_padding_mask)

            decoder_output = x
            for layer in self.decoder_layers:
                decoder_output = layer(decoder_output, encoder_output=x,
                                      cross_attn_mask=cross_attn_mask,
                                      cross_key_padding_mask=cross_key_padding_mask)

            output = self.output_norm(decoder_output)

        elif self.architecture == 'encoder':
            for layer in self.layers:
                x = layer(x, self_attn_mask=self_attn_mask, self_key_padding_mask=self_key_padding_mask)
            output = self.output_norm(x)

        elif self.architecture == 'decoder':
            causal_mask = AttentionMasking.create_causal_mask(x.shape[1])
            if self_attn_mask is not None:
                self_attn_mask = self_attn_mask + causal_mask
            else:
                self_attn_mask = causal_mask

            for layer in self.layers:
                x = layer(x, self_attn_mask=self_attn_mask, self_key_padding_mask=self_key_padding_mask)
            output = self.output_norm(x)

        return output

    def get_config(self):
        return {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'architecture': self.architecture
        }


class MultimodalMLA(nn.Module):
    def __init__(self, text_embed_dim, image_embed_dim, audio_embed_dim, 
                 hidden_dim=512, num_heads=8, num_layers=6, 
                 ffn_dim=2048, dropout=0.1):
        super(MultimodalMLA, self).__init__()
        
        self.text_embed_dim = text_embed_dim
        self.image_embed_dim = image_embed_dim
        self.audio_embed_dim = audio_embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)
        self.image_proj = nn.Linear(image_embed_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_embed_dim, hidden_dim)
        
        self.modality_weights = Tensor([1.0, 1.0, 1.0])
        
        self.cross_attention_layers = []
        for _ in range(num_layers):
            self.cross_attention_layers.append(CrossAttention(hidden_dim, num_heads, dropout))
        
        self.mla_layers = []
        for _ in range(num_layers):
            self.mla_layers.append(MLALayer(hidden_dim, num_heads, ffn_dim, dropout, use_cross_attention=False))
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, text_embeds, image_embeds, audio_embeds=None):
        text_proj = self.text_proj(text_embeds)
        image_proj = self.image_proj(image_embeds)
        
        weights = self.modality_weights.softmax(dim=0)
        
        x = text_proj
        for i, cross_attn in enumerate(self.cross_attention_layers):
            x, _ = cross_attn(x, image_proj)
            
            if audio_embeds is not None:
                audio_proj = self.audio_proj(audio_embeds)
                x, _ = cross_attn(x, audio_proj)
        
        for layer in self.mla_layers:
            x = layer(x)
        
        output = self.output_norm(x)
        
        return output
    
    def get_modality_weights(self):
        return self.modality_weights.softmax(dim=0)
    
    def get_config(self):
        return {
            'text_embed_dim': self.text_embed_dim,
            'image_embed_dim': self.image_embed_dim,
            'audio_embed_dim': self.audio_embed_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers
        }