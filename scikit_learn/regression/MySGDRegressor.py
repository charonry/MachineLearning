"""
SGDRegressor:线性回归-梯度下降 监督学习算法，
核心用途就是回归任务（预测连续值）
"""
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
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
# 3.创建和训练线性回归模型(梯度下降)
sgd_model = SGDRegressor()
sgd_model.fit(X_train_scaled, y_train)
print(f"权重系数{sgd_model.coef_}")
# 在线性回归模型中，•偏置（截距）•（bias）是模型参数之一，表示当所有特征值为0时，目标变量的期望值。
print(f"偏置值{sgd_model.intercept_}")
# 4.模型评估
y_predict = sgd_model.predict(X_test_scaled)
print(f"预测值：{y_predict}")
mse = mean_squared_error(y_test, y_predict)
print(f"梯度下降方程-均方误差：{mse}")
