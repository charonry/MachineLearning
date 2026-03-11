import math
import torch
from torch import nn
from torch.autograd import Variable


class MyEmbeddings(nn.Module):
    def __init__(self, vocab, d_model):
        # 类的初始化函数, 有两个参数, d_model: 指词嵌⼊的维度, vocab: 指词表的⼤⼩
        super(MyEmbeddings, self).__init__()
        # 调⽤nn中的预定义层Embedding, 获得⼀个词嵌⼊对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, input):
        # 参数input: 因为Embedding层是⾸层, 所以代表输⼊给模型的⽂本通过词汇映射后的张量
        return self.lut(input) * math.sqrt(self.d_model)


"""
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = nn.Embedding(1000, 512)
embr = emb(x)
print(embr.shape) 2*4*512
"""
x = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
myEmb = MyEmbeddings(1000, 512)
myEmbr = myEmb(x)

if __name__ == '__main__':
    print(myEmbr.shape, myEmbr)
