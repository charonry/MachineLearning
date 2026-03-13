"""
解码器:
根据编码器的结果以及上⼀次预测的结果, 对下⼀次可能出现的'值'进⾏特征表示.
"""
import copy
import torch
from torch import nn
from transformer.inputLayer.MyPositionalEncoding import result as pe_result
from transformer.encoderLayer.MyMultiHeadedAttention import clones
from transformer.encoderLayer.MyLayerNorm import MyLayerNorm
from transformer.encoderLayer.MyMultiHeadedAttention import MyMultiHeadedAttention
from transformer.encoderLayer.MyPositionwiseFeedForward import MyPositionwiseFeedForward
from transformer.encoderLayer.MyEncoder import en_result
from transformer.decoderLayer.MyDecoderLayer import MyDecoderLayer


class MyDecoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer:解码器层layer
        :param N:是解码器层的个数N
        """
        super(MyDecoder, self).__init__()
        # 克隆N个解码层放到编码器中
        self.layers = clones(layer, N)
        # 初始化规范层，作用在编码器最后
        self.norm = MyLayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


size = d_model = 512
d_ff = 64
head = 8
N = 8
dropout = 0.2
c = copy.deepcopy
self_atten = src_atten = MyMultiHeadedAttention(head, d_model, dropout)
ff = MyPositionwiseFeedForward(d_model, d_ff)
layer = MyDecoderLayer(size, c(self_atten), c(src_atten), c(ff), dropout)
myDecoder = MyDecoder(layer, N)
source_mask = target_mask = torch.zeros(2, 4, 4)
de_result = myDecoder(pe_result, en_result, source_mask, target_mask)

if __name__ == '__main__':
    print(de_result.shape)
    print(de_result)
