"""
规范化层:
所有深层⽹络模型都需要的标准⽹络层，因为随着⽹络层数的增加，通过多层的计算后参数可能开始出现过⼤或过⼩的情况，这样可能会
导致学习过程出现异常，模型可能收敛⾮常的慢. 因此都会在⼀定层数后接规范化层进⾏数值的规范化，使其特征数值在合理范围内
"""

from transformer.encoderLayer.MyPositionwiseFeedForward import pff_result
from torch import nn
import torch


class MyLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features: 示词嵌⼊的维度
        :param eps: ⾜够⼩的数, 在规范化公式的分⺟中出现,防⽌分⺟为0.默认是1e-6.
        """
        super(MyLayerNorm, self).__init__()
        # 根据features的形状初始化2个张量。使⽤nn.parameter封装，代表他们是模型的参数
        # 因为直接对上⼀层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因⼦，
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 求均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进⾏乘法操作，加上位移参数b2.
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


myLayerNorm = MyLayerNorm(512, eps=1e-6)
ln_result = myLayerNorm(pff_result)

if __name__ == '__main__':
    print(ln_result.shape)
    print(ln_result)
