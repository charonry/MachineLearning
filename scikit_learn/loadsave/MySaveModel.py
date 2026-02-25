import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
randomForest = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=3)
randomForest.fit(X_train, y_train)

# 保存模型
joblib.dump(randomForest, 'rfc_model.pkl')

# y_pred = randomForest.predict(X_test)
# print(f"正确值：{y_test}\n随机森林预测值：{y_pred}")
# accuracy = accuracy_score(y_test, y_pred)
# print(f'测试集准确率：{accuracy:.4f}')
# print('分类报告：\n', classification_report(y_test, y_pred, target_names=iris.target_names))
