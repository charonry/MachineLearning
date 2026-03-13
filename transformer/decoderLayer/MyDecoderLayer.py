"""
解码器层:
作为解码器的组成单元, 每个解码器层根据给定的输⼊向⽬标⽅向进⾏特征提取操作，即解码过程.
解码器层中的各个部分，如，多头注意⼒机制，规范化层，前馈全连接⽹络，⼦层连接结构都与编码器中的实现相同.
因此这⾥可以直接拿来构建解码器层
"""
import torch
from torch import nn
from transformer.inputLayer.MyPositionalEncoding import result as pe_result
from transformer.encoderLayer.MyMultiHeadedAttention import clones
from transformer.encoderLayer.MySublayerConnection import MySubLayerConnection
from transformer.encoderLayer.MyMultiHeadedAttention import MyMultiHeadedAttention
from transformer.encoderLayer.MyPositionwiseFeedForward import MyPositionwiseFeedForward
from transformer.encoderLayer.MyEncoder import en_result


class MyDecoderLayer(nn.Module):
    def __init__(self, size, self_atten, src_atten, feed_forward, dropout):
        """
        :param size: 词嵌入的维度
        :param self_atten: 多头自注意力机制对象 Q=K=V
        :param src_atten: 常规注意力机制对象 Q！=K=V
        :param feed_forward: 前馈全连接层
        :param dropout: 置零比例
        """
        super(MyDecoderLayer, self).__init__()
        self.size = size
        self.self_atten = self_atten
        self.src_atten = src_atten
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 克隆三个⼦层连接对象
        self.sublayer = clones(MySubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层输入张量
        :param memory: 编码器的寓意存储张量
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        """
        m = memory
        # 1.经历第一个子层：采用多头自注意力机制的子层
        # 采用target_mask，为了对⽬标数据进⾏遮掩
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, target_mask))
        # 2.经历第二个子层：采用常规注意力机制的子层。q是输⼊x; k，v是编码层输出 memory
        # 采用source_mask，进⾏源数据遮掩的原因并⾮是抑制信息泄漏，遮蔽掉对结果没有意义的字符⽽产⽣的注意⼒值
        x = self.sublayer[1](x, lambda x: self.src_atten(x, m, m, source_mask))
        # 3.经历第三个子层：前馈全连接⼦层，经过它的处理后就可以返回结果
        return self.sublayer[2](x, self.feed_forward)


size = d_model = 512
head = 8
d_ff = 64
dropout = 0.1
# src_attn和self_attn是同⼀个类.
self_atten = src_atten = MyMultiHeadedAttention(head, d_model, dropout)
# 实际中source_mask和target_mask并不相同
source_mask = target_mask = torch.zeros(2, 4, 4)
ff = MyPositionwiseFeedForward(d_model, d_ff)
myDecoderLayer = MyDecoderLayer(size, self_atten, src_atten, ff, dropout)
dl_result = myDecoderLayer(pe_result, en_result, source_mask, target_mask)

if __name__ == '__main__':
    print(dl_result.shape)
    print(dl_result)
