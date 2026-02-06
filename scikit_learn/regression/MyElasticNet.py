"""
弹性网络ElasticNet：它通过同时使用L1和L2正则化项，克服了Lasso和Ridge回归各自的一些局限性
监督学习算法，
核心作用：特别适用于特征数量远大于样本数量（p >> n）或者特征高度相关的情况
"""
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.构建数据
california = fetch_california_housing()
# 2.数据预处理
X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.3, random_state=20)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 3.创建和训练线性回归模型(弹性网络)
en_model = ElasticNet()
en_model.fit(X_train_scaled, y_train)
print(f"权重系数{en_model.coef_}")
# 在线性回归模型中，•偏置（截距）•（bias）是模型参数之一，表示当所有特征值为0时，目标变量的期望值。
print(f"偏置值{en_model.intercept_}")
# 4.模型评估
y_predict = en_model.predict(X_test_scaled)
print(f"预测值：{y_predict}")
mse = mean_squared_error(y_test, y_predict)
print(f"弹性网络方程-均方误差：{mse}")
