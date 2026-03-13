"""
子层连接结构:
输⼊到每个⼦层以及规范化层的过程中，还使⽤了残差链接（跳跃连接），我们把这⼀部分结构整体叫做⼦层连接（代表⼦层及其链接结构），
在每个编码器层中，都有两个⼦层。这两个⼦层加上周围的链接结构就形成了两个⼦层连接结构。
"""
import torch
from torch import nn
from transformer.encoderLayer.MyLayerNorm import MyLayerNorm
from transformer.inputLayer.MyPositionalEncoding import result as pe_result
from transformer.encoderLayer.MyMultiHeadedAttention import MyMultiHeadedAttention


class MySubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size:词维 dropout：置0比率
        super(MySubLayerConnection, self).__init__()
        # 实例规范化对象
        self.norm = MyLayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self, x, sublayer):
        # sublayer:该子层连接中子层函数我们⾸先对输出进⾏规范化，然后将结果传给⼦层处理，之后再对⼦层进⾏dropout操作，
        # x规范化、送入子层函数操作，结果dropout层，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))


mask = torch.zeros(2, 4, 4)
mmha = MyMultiHeadedAttention(8, 512, 0.1)
# 使⽤lambda获得⼀个函数类型的⼦层
sublayer = lambda x: mmha(x, x, x, mask)
mySubLayerConnection = MySubLayerConnection(512, 0.1)
sc_result = mySubLayerConnection(pe_result, sublayer)

if __name__ == '__main__':
    print(sc_result.shape)
    print(sc_result)
