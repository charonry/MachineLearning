"""
DecisionTree:决策树 监督学习算法，
用于分类和回归任务。
"""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target
# 1.数据预处理:# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 2.创建决策树分类器
decisionTree = DecisionTreeClassifier()
decisionTree.fit(X_train, y_train)
# 3.进行预测并评估模型
y_pred = decisionTree.predict(X_test)
print(f"正确值：{y_test}\n决策树预测值：{y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率：{accuracy:.4f}')
print('分类报告：\n', classification_report(y_test, y_pred, target_names=iris.target_names))
