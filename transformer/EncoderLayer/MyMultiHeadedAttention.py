"""
多头注意力机制：
我只有使⽤了⼀组线性变化层，即三个变换张量对Q，K，V分别进⾏线性变换，这些变换不会改变原有张量的尺⼨，
因此每个变换矩阵都是⽅阵，得到输出结果后，多头的作⽤才开始显现，
每个头开始从词义层⾯分割输出的张量，也就是每个头都想获得⼀组Q，K，V进⾏注意⼒机制的计算，
但是句⼦中的每个词的表示只获得⼀部分，也就是只分割了最后⼀维的词嵌⼊向量. 这就是所谓的多头，
将每个头的获得的输⼊送到注意⼒机制中, 就形成多头注意⼒机制。
作用：
这种结构设计能让每个注意⼒机制去优化每个词汇的不同特征部分，从⽽均衡同⼀种注意⼒机制可能产⽣的偏差，
让词义拥有来⾃更多元的表达，实验表明可以从⽽提升模型效果
"""

import copy
from torch import nn
from transformer.inputLayer.MyPositionalEncoding import result as pe_result
from transformer.EncoderLayer.MyAttention import MyAttention
import torch


def clones(module, N):
    """
    ⾸先需要定义克隆函数, 因为在多头注意⼒机制的实现中, ⽤到多个结构相同的线性层
    将他们⼀同初始化在⼀个⽹络层列表对象中. 之后的结构中也会⽤到该函数
    :param module: 要克隆的⽬标⽹络层,
    :param N: 克隆的数量
    :return: 通过for循环对module进⾏N次深度拷⻉, 使其每个module成为独⽴的层
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ⽤⼀个类来实现多头注意⼒机制的处理
class MyMultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        :param head: 头数
        :param embedding_dim: 表词嵌⼊的维度
        :param dropout: dropout操作时置0⽐率
        """
        super(MyMultiHeadedAttention, self).__init__()
        # 判断head是否能被是embedding_dim整除
        assert embedding_dim % head == 0
        # 给每个头分配等量的词特征.也就是embedding_dim/head个.
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        # 获取线性层 Linear实例化，内部变换矩阵是embedding_dim x embedding_dim（方阵），使⽤clones函数克隆四个
        # 为什么是四个呢，这是因为在多头注意⼒中，Q，K，V各需要⼀个，最后拼接的矩阵还需要⼀个，因此⼀共是四个.
        self.linear = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # 初始化参数
        self.mask = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 代表是多头中第n个头
            mask = mask.unsqueeze(1)
        # 他是query尺⼨的第1个数字，代表有多少条样本
        batch_size = query.size(0)
        """
        # 之后就进⼊多头处理环节
        zip将网络层和输入数据连接在一起，模型的输出利用view和transpose维度重塑
        transpose为了让代表句⼦⻓度维度和词向量维度能够相邻，这样注意⼒机制才能找到词义与句⼦位置的关系
        """
        query, key, value = [module(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for module, x in
                             zip(self.linear, (query, key, value))]
        # 将每个头的输⼊传⼊到attention中，
        x, self.attn = MyAttention(query, key, value, mask=mask, dropout=self.dropout)
        # 每个头的计算结果是4维张量，需要进行形状转换
        # 前面已经转置过需要重新转置回来：经历transpose之后需要contiguous才能重新调用view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        # 最终将x输入线性层列表中最后一个线性层进行处理，得到最终多头注意力结构输出
        return self.linear[-1](x)


myMultiHeadedAttention = MyMultiHeadedAttention(8, 512, 0.2)
query = key = value = pe_result
mask = torch.zeros(2, 4, 4)
mha_result = myMultiHeadedAttention(query, key, value, mask=mask)
if __name__ == '__main__':
    print(mha_result.shape)
    print(mha_result)
