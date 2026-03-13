"""
注意力机制：
注意⼒计算规则:需要三个指定的输⼊Q(query), K(key), V(value), 然后通过公式得到注意⼒的计算结果,
这个结果代表query在key和value作⽤下的表示。
自注意力机制：
注意⼒机制是注意⼒计算规则能够应⽤的深度学习⽹络的载体, 除了注意⼒计算规则外,
还包括⼀些必要的全连接层以及相关张量处理, 使其与应⽤⽹络融为⼀体. 使⽤⾃注意⼒
计算规则的注意⼒机制称为⾃注意⼒机制.
"""
import math
import torch
import torch.nn.functional as F
from transformer.inputLayer.MyPositionalEncoding import result as pe_result


def MyAttention(query, key, value, mask, dropout=None):
    """
    query, key, value 注意力三个输入张量
    mask: 掩码张量
    dropout是nn.Dropout层的实例化对象, 默认为None
    """
    # 取query的最后⼀维的⼤⼩，代表词维
    d_k = query.size(-1)
    # 按照注意⼒公式, 将query与key的转置相乘, 这⾥⾯key是将最后两个维度进⾏转置, 再除以缩放系数根号下d_k, 这种计算⽅法也称为缩放点积注意⼒计算
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 接着判断是否使⽤掩码张量
    if mask is not None:
        # 使⽤tensor的masked_fill⽅法,将掩码张量和scores张量每个位置⼀⼀⽐较, 如果掩码张量处为0替换为-1e9
        scores = scores.masked_fill(mask == 0, -1e9)
    # 注意⼒张量:对scores的最后⼀维进⾏softmax操作 第⼀个参数是softmax对象, 第⼆个是⽬标维度.
    p_attn = F.softmax(scores, dim=-1)
    # 之后判断是否使⽤dropout进⾏随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力输出（上下文向量） 和 注意⼒权重
    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_result
mask = torch.zeros(2, 4, 4)
attn, p_attn = MyAttention(query, key, value, mask=mask)

if __name__ == '__main__':
    print(attn.shape)
    print("*" * 50)
    print(attn)
    print("*" * 50)
    print(p_attn.shape)
    print("*" * 50)
    print(p_attn)
