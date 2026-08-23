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
    attn_mask = torch.ones((b, max_lable_len, max_feat_len), device=device)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:]
    return attn_mask.to(torch.bool)


if __name__ == "__main__":
    print(torch.__version__)