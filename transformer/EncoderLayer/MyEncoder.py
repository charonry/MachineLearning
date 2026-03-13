"""
编码器:
编码器⽤于对输⼊进⾏指定的特征提取过程, 也称为编码, 由N个编码器层堆叠⽽成
"""
import copy
import torch
from torch import nn
from transformer.inputLayer.MyPositionalEncoding import result as pe_result
from transformer.encoderLayer.MyEncoderLayer import MyEncoderLayer
from transformer.encoderLayer.MyMultiHeadedAttention import clones, MyMultiHeadedAttention
from transformer.encoderLayer.MyPositionwiseFeedForward import MyPositionwiseFeedForward
from transformer.encoderLayer.MyLayerNorm import MyLayerNorm


class MyEncoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 代表编码层
        :param N: 编码器中N个编码层
        """
        super(MyEncoder, self).__init__()
        # 克隆N个编码层放到编码器中
        self.layers = clones(layer, N)
        # 初始化规范层，作用在编码器最后
        self.norm = MyLayerNorm(layer.size)

    def forward(self, x, mask):
        """
        :param x: 上一层输出的张量
        :param mask: 掩码张量
        """
        # 依次进入编码层进行处理，最终进行规范化层输出
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


size = d_model = 512
d_ff = 64
head = 8
mask = torch.zeros(2, 4, 4)
N = 8
dropout = 0.2
attn = MyMultiHeadedAttention(head, d_model)
ff = MyPositionwiseFeedForward(d_model, d_ff)
c = copy.deepcopy
layer = MyEncoderLayer(size, c(attn), c(ff), dropout)
myEncoder = MyEncoder(layer, N)
en_result = myEncoder(pe_result, mask)

if __name__ == '__main__':
    print(en_result.shape)
    print(en_result)
