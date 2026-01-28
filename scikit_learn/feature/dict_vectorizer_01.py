from sklearn.feature_extraction import DictVectorizer

# 示例数据，列表中的每个元素都是一个字典，代表一个样本
data = [
    {'age': 25, 'city': 'New York', 'income': 50000},
    {'age': 30, 'city': 'Boston', 'income': 65000},
    {'age': 35, 'city': 'New York', 'income': 75000}
]

# 初始化 DictVectorizer
dict_vectorizer = DictVectorizer(sparse=True)
# 学习并转换数据
x_dict = dict_vectorizer.fit_transform(data)
print(f"特征名称:\n{dict_vectorizer.get_feature_names_out()}")
print(f"编码后的特征矩阵:\n{x_dict}")
