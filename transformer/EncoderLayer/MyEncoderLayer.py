"""
编码器层的作⽤:
作为编码器的组成单元, 每个编码器层完成⼀次对输⼊的特征提取过程, 即编码过程.
"""
import torch
from transformer.encoderLayer.MyMultiHeadedAttention import clones, MyMultiHeadedAttention
from transformer.encoderLayer.MySublayerConnection import MySubLayerConnection
from transformer.encoderLayer.MyPositionwiseFeedForward import MyPositionwiseFeedForward
from transformer.inputLayer.MyPositionalEncoding import result as pe_result
from torch import nn


class MyEncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forword, dropout=0.1):
        super(MyEncoderLayer, self).__init__()
        """
        :param size: 词嵌入维度
        :param attn: 多头注意力机制子层实例化对象
        :param feed_word: 前馈全连接层实例化对象
        :param dropout: 置零比例
        """
        self.size = size
        self.attn = attn
        self.feed_forword = feed_forword
        # 编码器层有2个子层连接结构
        self.sublayer = clones(MySubLayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        :param x: 上一层的传入张量
        :param mask: 掩码张量
        """
        # 首先让x经历第一个子层连接结构，包含一个多头注意力机制子层
        # 再让张量经历第二个子层连接结构，包含一个前馈全连接网络
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forword)


size = d_model = 512
head = 8
d_ff = 64
x = pe_result
dropout = 0.1
mask = torch.zeros(2, 4, 4)
attn = MyMultiHeadedAttention(head, d_model, dropout)
feed_forword = MyPositionwiseFeedForward(d_model, d_ff, dropout)
myEncoderLayer = MyEncoderLayer(size, attn, feed_forword, dropout)
el_result = myEncoderLayer(pe_result, mask)

if __name__ == '__main__':
    print(el_result.shape)
    print(el_result)
