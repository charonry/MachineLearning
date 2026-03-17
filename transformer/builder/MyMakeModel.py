"""
Tansformer模型构建
"""
import copy
import torch
import torch.nn as nn
from transformer.encoderLayer.MyMultiHeadedAttention import MyMultiHeadedAttention
from transformer.encoderLayer.MyPositionwiseFeedForward import MyPositionwiseFeedForward
from transformer.inputLayer.MyPositionalEncoding import MyPositionalEncoding
from transformer.inputLayer.MyInputEmbedding import MyEmbeddings
from transformer.builder.MyEncoderDecoder import MyEncoderDecoder
from transformer.encoderLayer.MyEncoder import MyEncoder
from transformer.encoderLayer.MyEncoderLayer import MyEncoderLayer
from transformer.decoderLayer.MyDecoder import MyDecoder
from transformer.decoderLayer.MyDecoderLayer import MyDecoderLayer
from transformer.outputLayer.MyGenerator import MyGenerator


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    :param source_vocab: 源数据词汇总数
    :param target_vocab: 目标数据的词汇总数
    :param N: 编码器和解码器堆叠的层数
    :param d_model: 词汇的维度
    :param d_ff: 前馈全连接层中变换矩阵维度
    :param head: 多头注意力机制中头数
    :param dropout: 置零比率
    """
    c = copy.deepcopy

    # 实例化多头注意力类
    attn = MyMultiHeadedAttention(head, d_model, dropout)

    # 实例化前馈全连接层的网络对象
    ff = MyPositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码器对象
    position = MyPositionalEncoding(d_model, dropout)

    # 实例化模型model，利用MyEncoderDecoder类
    # 编码器结构含有2个子层，attention和前馈全连接层
    # 解码器结构含有3个子层，2个attention和前馈全连接层
    model = MyEncoderDecoder(
        MyEncoder(MyEncoderLayer(d_model, c(attn), c(ff), dropout), N),
        MyDecoder(MyDecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(MyEmbeddings(source_vocab, d_model), position),
        nn.Sequential(MyEmbeddings(target_vocab, d_model), position),
        MyGenerator(d_model, target_vocab)
    )

    # 初始化模型参数，判断模型维度>1将矩阵转换为一个服从正态分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal(p)

    return model


if __name__ == '__main__':
    source_vocab = target_vocab = 11
    N = 6
    res = make_model(source_vocab, target_vocab, N)
    print(res)
