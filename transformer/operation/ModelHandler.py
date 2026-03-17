"""
第二步：获得transformer模型及其优化器和损失函数
"""
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from transformer.builder.MyMakeModel import make_model


class SimpleLossCompute:
    """兼容新版 PyTorch 的损失计算类"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # 使用 item() 替代 data[0]，兼容新版 PyTorch
        return loss.item() * norm


V = 11
# 模型实例化对象
model = make_model(V, V, N=2)
# 优化器
model_optimizer = get_std_opt(model)
# 标签平滑对象
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# 损失函数
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
