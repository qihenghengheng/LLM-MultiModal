import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===================== 1. Positional Encoding 位置编码 =====================
class PositionalEncoding(nn.Module):
    """位置编码，提供输入向量时序信息"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ===================== 2. Scaled Dot‑Product Attention 缩放点积注意力 =====================
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q,k,v: [batch, head, seq_len, d_k]
    mask: True代表需要mask遮蔽
    """
    d_k = q.size(-1)
    attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_score = attn_score.masked_fill(mask, -1e9)
    attn_weight = F.softmax(attn_score, dim=-1)
    output = torch.matmul(attn_weight, v)
    return output, attn_weight


# ===================== 3. Multi‑Head Attention 多头自注意力机制 =====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 线性投影 +分头
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        out, attn = scaled_dot_product_attention(q, k, v, mask)
        # concat多头
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        return self.wo(out), attn


# ===================== 4. Feed Forward 前馈神经网络层 =====================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# ===================== 5. Encoder Layer 编码器层（原图N×堆叠） =====================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # ========== 原版论文 Post‑LN：注意力计算 → dropout → Add残差 → Norm ==========
        attn_out, _ = self.mha(x, x, x, src_mask)
        x = self.norm1(x + self.drop1(attn_out))   # Add & Norm

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))    # Add & Norm
        return x


# ===================== 6. Decoder Layer 解码器层（原图N×堆叠） =====================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads)  # Masked Multi‑Head Attention 带因果掩码
        self.cross_mha = MultiHeadAttention(d_model, num_heads)   # Multi‑Head Attention 交叉注意力
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_memory, tgt_mask=None, memory_mask=None):
        # 第一层：Masked Multi‑Head Attention
        attn1, _ = self.masked_mha(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(attn1))

        # 第二层：Multi‑Head 交叉注意力 Q来自decoder，K/V来自encoder输出
        attn2, _ = self.cross_mha(x, enc_memory, enc_memory, memory_mask)
        x = self.norm2(x + self.drop2(attn2))

        # 第三层 FeedForward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.drop3(ffn_out))
        return x


# ===================== 7. Transformer完整模型 =====================
class TransformerOriginal(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_layers=6, num_heads=8,
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Input Embedding / Output Embedding：词嵌入向量
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # N层编码器堆叠
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        # N层解码器堆叠
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # Linear全连接层，映射到词表
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        """
        src_tokens: [batch, src_seq_len] 源输入token id
        tgt_tokens: [batch, tgt_seq_len] 目标输入token id（shifted right）
        return logits [batch, tgt_seq_len, tgt_vocab_size]
        """
        # Embedding * sqrt(d_model) + Positional Encoding
        src_x = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        src_x = self.pos_encoding(src_x)

        tgt_x = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
        tgt_x = self.pos_encoding(tgt_x)

        # Encoder前向传播 N×
        memory = src_x
        for enc_layer in self.encoder:
            memory = enc_layer(memory, src_mask)

        # Decoder前向传播 N×
        dec_out = tgt_x
        for dec_layer in self.decoder:
            dec_out = dec_layer(dec_out, memory, tgt_mask, src_mask)

        # Linear + Softmax（训练时返回logits，softmax放在loss计算）
        logits = self.final_linear(dec_out)
        return logits


# ===================== 工具函数：生成因果掩码（Masked Multi‑Head Attention用） =====================
def generate_causal_mask(seq_len):
    """上三角掩码，防止解码器看到未来token"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0) # shape [1, 1, seq_len, seq_len]


# ===================== 测试运行 =====================
if __name__ == "__main__":
    src_vocab = 10000
    tgt_vocab = 10000
    batch = 2
    src_seq_len = 15
    tgt_seq_len = 12

    model = TransformerOriginal(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048
    )

    src = torch.randint(low=0, high=src_vocab, size=(batch, src_seq_len))
    tgt = torch.randint(low=0, high=tgt_vocab, size=(batch, tgt_seq_len))
    tgt_mask = generate_causal_mask(tgt_seq_len)

    output_logits = model(src, tgt, tgt_mask=tgt_mask)
    print(f"输出logits shape: {output_logits.shape}")
    # torch.Size([2, 12, 10000]) [batch, tgt_len, vocab_size]
