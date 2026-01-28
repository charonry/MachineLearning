from sklearn.preprocessing import OneHotEncoder

# 示例类别数据
data = [['Red'], ['Blue'], ['Green'], ['Blue'], ['Red']]

# 初始化 OneHotEncoder
# sparse_output-False（默认）:密集矩阵,True:稀疏矩阵（节约压缩空间）
# handle_unknown='ignore' 防止遇到未知类别时报错
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# 学习并转换数据
# fit() ：用于从训练数据生成学习模型参数
# transform()：从fit()方法生成的参数，应用于模型以生成转换数据集。
# fit_transform()：在同一数据集上组合fit()和transform() api
X_encoder = encoder.fit_transform(data)
# OneHotEncoder 的使用必须遵循「先拟合（fit）→ 再转换 / 获取属性」
print(f"特征名称:\n{encoder.get_feature_names_out()}")
print(f"编码后的特征矩阵:\n{X_encoder}")

new_data = [['Blue'], ['Red'], ['Yellow']]
X_new_encoder = encoder.transform(new_data)
print(f"新数据编码后的特征矩阵:\n{X_new_encoder}")
