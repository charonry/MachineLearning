"""
掩码张量：
在transformer中, 掩码张量的主要作⽤在应⽤attention时，有⼀些⽣成的attention张量中的值计算有可能
已知了未来信息⽽得到的，未来信息被看到是因为训练时会把整个输出结果都⼀次性进⾏Embedding，但是理论
上解码器的的输出却不是⼀次就能产⽣最终结果的，⽽是⼀次次通过上⼀次结果综合得出的，因此，未来的信息
可能被提前利⽤. 所以，我们会进⾏遮掩.
"""
import numpy as np
import torch


def subsequent_mask(size):
    # 定义掩码张量的形状
    attn_shape = (1, size, size)
    # ⽤np.ones⽅法向这个形状中添加1元素, 形成上三⻆阵,
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 最后将numpy类型转化为torch中的tensor, 内部做⼀个1 - 的操作:做了⼀个三⻆阵的反转
    return torch.from_numpy(1 - mask)


sm = subsequent_mask(5)
if __name__ == '__main__':
    print("subsequent_mask:", sm)
