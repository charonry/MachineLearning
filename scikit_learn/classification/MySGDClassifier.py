"""
SGDClassifier:随机梯度下降分类器 监督学习
高效处理大规模、超大规模数据集的线性分类任务
通过切换loss参数，模拟逻辑回归、线性SVM
作为分类任务的基线模型。
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

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
# 3.创建和训练SGDClassifier模型
sgd_clf = SGDClassifier(
    loss="log_loss",  # 损失函数：选择 log_loss，模拟逻辑回归效果
    max_iter=1000,  # 最大迭代次数
    tol=1e-3,  # 收敛阈值
    random_state=42  # 固定随机种子，保证训练结果可复现
)
# 训练模型（传入标准化后的训练数据）
sgd_clf.fit(X_train_scaled, y_train)
# 4.进行预测并评估模型
y_pred = sgd_clf.predict(X_test_scaled)
print(f"正确值：{y_test}\n随机梯度下降分类器预测值：{y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率：{accuracy:.4f}')
print('分类报告：\n', classification_report(y_test, y_pred, target_names=iris['target_names']))
