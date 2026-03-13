"""
输出部分:
输出部分包含:线性层、softmax层
"""
from torch import nn
from torch.nn import functional as F
from transformer.decoderLayer.MyDecoder import de_result


class MyGenerator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        :param d_model:词嵌⼊维度
        :param vocab_size:词表⼤小
        """
        super(MyGenerator, self).__init__()
        # 使⽤nn中的预定义线性层进⾏实例化
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        log_softmax就是对softmax的结果⼜取了对数, 因为对数函数是单调递增函数,
        """
        return F.softmax(self.project(x), dim=-1)


d_model = 512
vocab_size = 1000
myGenerator = MyGenerator(d_model, vocab_size)
gen_result = myGenerator(de_result)

if __name__ == '__main__':
    print(gen_result.shape)
    print(gen_result)
