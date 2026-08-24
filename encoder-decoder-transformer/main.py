import numpy as np
import torch
import torch.nn as nn

def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    :param b: batch size
    :param max_len: 输入序列最大长度
    :param feat_lens: 输入序列长度
    :param device: 设备
    :return: shape(b, max_len, max_len), True表示该位置需要遮挡的Encoder长度掩码
    """
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)

def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """
    :param b: batch size
    :param max_len:  输入序列最大长度
    :param device:  设备
    :return:  shape(b, max_len, max_len), True表示该位置需要遮挡的因果掩码
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)

def get_enc_dec_mask(b: int, max_lable_len: int, max_feat_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    :param b: batch size
    :param max_lable_len: 输出序列最大长度
    :param max_feat_len: 输入序列最大长度
    :param feat_lens: 输入序列长度
    :param device: 设备
    :return: shape(b, max_lable_len, max_feat_len), True表示位置需要遮挡的交叉注意力掩码
    """
    attn_mask = torch.zeros((b, max_lable_len, max_feat_len), device=device)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)

def pos_sinusoid_embedding(seq_len: int, d_model:int) -> torch.Tensor:
    embedding = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embedding[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embedding.float()

class PoswiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float=0.):
        super(PoswiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.dropout(out)


class MultiHeadAttention(nn.Module):
    def __init__ (self, d_k, d_v, d_model, num_heads, p=0.):
        """
        :param d_k: 每个注意力头的 Key/Query 维度
        :param d_v: 每个注意力头的 Valse维度
        :param d_model:  输入/输出的总维度
        :param num_heads: 注意力头数
        :param p: 
        """
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)

        # 四个线性投影矩阵：Q、K、V投影和输出投影
        self.W_Q = nn.Linear(self.dmodel, self.d_k * num_heads)
        self.W_K = nn.Linear(self.dmodel, self.d_k * num_heads)
        self.W_V = nn.Linear(self.dmodel, self.d_v * num_heads)
        self.W_out = nn.Linear(self.d_v * num_heads, self.d_model)

        # 权重初始化
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: (batch, q_len, d_model)
        :param K: (batch, q_len, d_model)
        :param V: (batch, v_len, d_model)
        :param attn_mask: (batch, q_len, q_len)
        :return:
        """
        N = Q.size(0)
        q_len, k_len, = Q.size(1), K.size(1)
        d_k, d_v, num_heads = self.d_k, self.d_v, self.num_heads

        # 线性投影 + 拆分多头
        # (batch, q_len, d_model) -> (batch, q_len, num_heads*d_k)
        Q = self.W_Q(Q).view(N, q_len, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, k_len, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, k_len, num_heads, d_v).transpose(1, 2)

        # 广播Mask到head维度
        if attn_mask is not None:
            assert attn_mask.size == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1).bool()

        # 相乘
        sorces = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            sorces.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(sorces, dim=-1)
        attns = self.dropout(attns)

        output = torch.matmul(attns, V)
        output = output.transpose(1, 2).contiguous().reshape(V, q_len, num_heads * d_v)
        output = self.W_out(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, dim: int, n: int, dff: int, dropout_posffn: float, dropout_attn: float):
        assert dim % n == 0
        super(EncoderLayer, self).__init__()
        hdim = dim // n

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim, dff, dropout_posffn)

    def forward(self, enc_in, attn_mask):
        residual = enc_in
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        out = self.norm1(context + residual)

        residual = out
        out = self.poswise_ffn(out)
        out = self.norm2(out + residual)
        return out

class Encoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn, num_layers, enc_dim, num_heads, dff, tgt_len):
        super(Encoder, self).__init__()
        self.tgt_len = tgt_len
        # 固定的正弦位置编码
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, enc_dim), freeze=True)
        self.emb_dropout = dropout_emb
        self.layers = nn.ModuleList([EncoderLayer(enc_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in range(num_layers)])

    def forward(self, X, X_lens, mask=None):
        # X = (batch, seq_len, d_model)
        seq_len = X.size(1)
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device))
        out = self.emb_dropout(out)
        for layer in self.layers:
            out = layer(out, mask)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, dim: int, n: int, dff: int, dropout_posffn: float, dropout_attn: float):
        assert dim % n == 0
        super(DecoderLayer, self).__init__()
        hdim = dim // n

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.poswise_ffn = PoswiseFFN(dim, dff, p=dropout_posffn)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)
        self.dec_enc_attn = MultiHeadAttention(hdim, hdim, dim, n, dropout_attn)

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask):
        """
        :param dec_in: (batch, dec_len, d_model) decoder当前层输入
        :param dec_out: (batch, enc_len, d_model) encoder输出
        :param dec_mask: (batch, dec_len, dec_len) causal mask
        :param dec_enc_mask: (batch, dec_len, enc_len) cross attention mask
        """
        residual = dec_in
        context = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_out = self.norm1(residual + context)

        residual = dec_out
        context = self.dec_enc_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        dec_out = self.norm2(residual + context)

        residual = dec_out
        dec_out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(dec_out + residual)
        return dec_out

class Decoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn,
                 num_layers, dec_dim, num_heads, dff, tgt_len, tgt_vocab_size):
        super(Decoder, self).__init__()
        # Word Embedding：将 token ID 映射为 d_model 维向量
        self.tgt_emb = nn.Embedding(tgt_vocab_size, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)
        # 固定正弦位置编码
        self.pos_emb = nn.Embedding.from_pretrained(
            pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn)
             for _ in range(num_layers)]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask):
        # labels: (batch, dec_len) token ID 序列
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)
        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out

class Transformer(nn.Module):
    def __init__(self, frontend: nn.Module, encoder: Encoder, decoder: Decoder, dec_out_dim: int, vocab: int):
        """
        :param frontend: 输入特征变换
        :param dec_out_dim: decoder输出维度=d_model
        :param vocab: 目标词表大小
        """
        super(Transformer, self).__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)

    def forward(self, X: torch.tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        """
        X:      (batch, enc_len, fbank_dim)  输入特征序列
        X_lens: (batch,)                      每个样本的实际输入长度
        labels: (batch, dec_len)              目标 token ID 序列
        Returns: logits (batch, dec_len, vocab_size)
        """
        X_lens, labels = X_lens.long(), labels.long()
        b, device = X.size(0), X.device

        # Frontend + Encoder（对应第 3 节 Encoder 结构）
        out = self.frontend(X)
        max_feat_len = out.size(1)
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)

        # Decoder（对应第 4 节 Decoder 结构）
        max_label_len = labels.size(1)
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)


if __name__ == "__main__":
    print(torch.__version__)