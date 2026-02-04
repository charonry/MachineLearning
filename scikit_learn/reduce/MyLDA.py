"""
LDA:线性判别分析-有监督
主要用于分类和降维。
"""
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris['data'], iris['target']
print(f"原始数据形状:{X.shape}")
standard_scaler = StandardScaler()
X_scaler = standard_scaler.fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaler, y)
print(f"降维后数据形状:{X_lda.shape}")
print(f"降维后数据形状:{X_lda}")
