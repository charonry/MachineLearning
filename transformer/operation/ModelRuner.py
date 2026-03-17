"""
第三步：运行模型进行训练和评估
"""
from pyitcast.transformer_utils import run_epoch
from transformer.operation.DataGenerator import data_generator
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


if __name__ == "__main__":
    run(model, loss)
