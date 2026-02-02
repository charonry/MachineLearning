from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

"""
f_classif 分类
计算每个特征与目标变量之间的 ANOVA F值。适用于连续特征和分类目标。默认选项。
chi2 分类
卡方检验。计算每个特征与目标变量之间的卡方统计量。适用于非负的特征（如词频、布尔特征）。
mutual_info_classif 分类
互信息。衡量特征和目标变量之间的非线性关系。非常强大，但计算成本更高。
f_regression 回归
计算每个特征与目标变量之间的 F值（线性回归模型的简单线性回归）。
mutual_info_regression 回归
互信息的回归版本，同样用于捕捉非线性关系。
"""
X, y = load_iris(return_X_y=True)
selector = SelectKBest(score_func=f_classif, k=3)
X_new = selector.fit_transform(X, y)
print(f"原始数据形状: {X.shape}")
print(f"筛选后特征数据形状: {X_new.shape}")
print(f"特征得分: {selector.scores_}")
