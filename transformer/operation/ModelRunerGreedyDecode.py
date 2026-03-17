"""
第四步：使用模型进行贪婪解码：就是改写第三步中run方法
贪婪解码是每次预测都选择概率最大的结果作为输出（不一定是全局最优性，但是拥有最高的执行效率）
"""
import torch
from pyitcast.transformer_utils import run_epoch
from transformer.operation.DataGenerator import data_generator
from pyitcast.transformer_utils import greedy_decode
from transformer.operation.ModelHandler import model, loss

V = 11


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        # 首先进入训练模式，所有参数会更新
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)
        # 训练结束后进入评估模式，所有参数固定不变
        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)

    # 模型训练结束之后进入评估模式
    model.eval()

    source = torch.LongTensor([[1, 2, 5, 3, 4, 6, 7, 8, 9, 10]])
    # 初始化一个掩码张量，全1代表没有遮掩
    source_mask = torch.ones(1, 1, 10)
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == "__main__":
    run(model, loss)
