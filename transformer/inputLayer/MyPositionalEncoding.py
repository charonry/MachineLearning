"""
位置编码器
"""
import math
import torch
from torch import nn
from MyInputEmbedding import myEmbr


class MyPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # d_model: 词嵌⼊维度,dropout: 置0⽐率, max_len: 每个句⼦的最⼤⻓度
        super(MyPositionalEncoding, self).__init__()
        # 实例化nn中预定义的Dropout层, 并将dropout传⼊其中
        self.dropout = nn.Dropout(p=dropout)
        # 初始化⼀个位置编码矩阵, 它是⼀个0阵，矩阵的⼤⼩是max_len x d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化⼀个绝对位置矩阵,使⽤unsqueeze⽅法拓展向量维度使其成为矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        # 经典初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(div_term * position)
        pe[:, 1::2] = torch.cos(div_term * position)
        # pe现在是⼀个⼆维矩阵，要想和embedding的输出（⼀个三维张量）相加，就必须拓展⼀个维度
        pe = pe.unsqueeze(0)
        # 最后把pe位置编码矩阵注册成模型的buffer
        # buffer:对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进⾏更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数⼀同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


myPositionalEncoding = MyPositionalEncoding(512, 0.1, 60)
x = myEmbr
result = myPositionalEncoding(x)
if __name__ == "__main__":
    print(x.shape)
    print("*" * 50)
    print(result.shape)
    print("*" * 50)
    print(result)
