"""
RandomForest:随机森林 监督学习算法，
用于分类和回归任务。
"""
# 1.基础随机森林
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
# 1.数据预处理:# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 2.创建随机森林分类器
randomForest = RandomForestClassifier(n_estimators=100, oob_score=True, max_depth=3)
randomForest.fit(X_train, y_train)
# 3.进行预测并评估模型
y_pred = randomForest.predict(X_test)
print(f"正确值：{y_test}\n随机森林预测值：{y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率：{accuracy:.4f}')
print('分类报告：\n', classification_report(y_test, y_pred, target_names=iris.target_names))
"""
# 2.随机森林-超参数调优
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# 1,加载数据
iris = load_iris()
X = iris.data
y = iris.target
# 2,数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
# 创建GridSearch对象
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5折交叉验证
    scoring='accuracy',
    n_jobs=-1  # 使用所有CPU核心并行计算
)
# 在训练数据上执行网格搜索 (警告：这可能很耗时！)
grid_search.fit(X_train, y_train)
# 输出最佳参数和最佳分数
print("最佳参数: ", grid_search.best_params_)
print("最佳交叉验证分数: ", grid_search.best_score_)
# 使用最佳参数的模型进行预测
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)
print("调优后测试集准确率: ", accuracy_score(y_test, y_pred_best))