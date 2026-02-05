"""
逻辑回归:监督学习算法，
以二分类任务为主，可扩展到多分类，还能预测类别概率。
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target
# 1.数据预处理:# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 2.数据标准化:消除不同特征量纲的影响
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
# 使用训练集的参数转换测试集
X_test_scaled = scaler.transform(X_test)
# 3.,创建和训练LogisticRegression模型
logistc = LogisticRegression()
logistc.fit(X_train_scaled, y_train)  # 使用训练数据拟合（训练）模型
# 4.进行预测并评估模型
y_pred = logistc.predict(X_test_scaled)
print(f"正确值：{y_test}\n逻辑回归预测值：{y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率：{accuracy:.4f}')
print('分类报告：\n', classification_report(y_test, y_pred, target_names=iris['target_names']))
