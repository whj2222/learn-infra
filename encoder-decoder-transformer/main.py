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


if __name__ == "__main__":
    print(torch.__version__)