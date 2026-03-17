"""
第一步：构建数据集生成器
"""

from pyitcast.transformer_utils import Batch
import numpy as np
import torch
from torch.autograd import Variable


def data_generator(V, batch_size, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(low=0, high=V, size=(batch_size, 10))).long()
        # 数据第一列设置1，为起始标志
        data[:, 0] = 1
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        yield Batch(source, target)


V = 11
batch_size = 20
num_batch = 30

if __name__ == "__main__":
    res = data_generator(V, batch_size, num_batch)
    print(res)
