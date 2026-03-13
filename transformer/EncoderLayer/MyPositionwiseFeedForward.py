"""
前馈全连接层：
在Transformer中前馈全连接层就是具有两层线性层的全连接⽹络
考虑注意⼒机制可能对复杂过程的拟合程度不够, 通过增加两层⽹络来增强模型的能⼒.
"""
from transformer.EncoderLayer.MyMultiHeadedAttention import mha_result
import torch
from torch import nn
from torch.nn import functional as F


class MyPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model:词维，也是2个线性层的输入输出 d_ff：第一个线性层输出=第二个线性层输入
        super(MyPositionwiseFeedForward, self).__init__()

        # 2个线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


myPositionwiseFeedForward = MyPositionwiseFeedForward(512, 64, 0.1)
pff_result = myPositionwiseFeedForward(mha_result)

if __name__ == '__main__':
    print(pff_result.shape)
    print(pff_result)
