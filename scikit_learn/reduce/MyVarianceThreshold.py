import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

X, y = load_iris(return_X_y=True)
feature_variances = np.var(X, axis=0)
feature_names = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]

print("鸢尾花数据 4 个特征的总体方差：")
for name, var in zip(feature_names, feature_variances):
    print(f"{name}：{var:.4f}")

selector = VarianceThreshold(threshold=0.5)
X_new = selector.fit_transform(X)
print("\n" + "-" * 50)
print(f"threshold=0.5 时，保留的特征索引：{selector.get_support(indices=True)}")
print(f"原始数据形状: {X.shape}")
print(f"筛选后特征数据形状: {X_new.shape}")
print(f"被剔除的特征：{[feature_names[i] for i in range(len(feature_names)) if not selector.get_support()[i]]}")
