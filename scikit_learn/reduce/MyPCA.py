"""
PCA:主成分分析算法 无监督学习 默认采用的是样本协方差
1.数据标准化：根据各列的均值方差获得标准化矩阵X∗
2.计算的协方差矩阵C
3.协方差矩阵C的特征值
4.对求应的单位特征向量
"""
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris['data'], iris['target']
print(f"原始数据形状:{X.shape}")
standard_scaler = StandardScaler()
X_scaler = standard_scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaler)
print(f"降维后数据形状:{X_pca.shape}")
print(f"降维后数据形状:{X_pca}")
