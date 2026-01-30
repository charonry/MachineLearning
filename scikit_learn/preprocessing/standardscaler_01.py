import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'age': [25, 30, 34, 45, 60, 30, 15],
    'salary': [50000, 54000, 60000, 54000, 100000, 40000, 20000],
    'country': ['USA', 'UK', 'China', 'USA', 'India', 'China', 'UK'],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'F']
}

df = pd.DataFrame(data)
age_mean = df['age'].mean()
salary_mean = df['salary'].mean()
# 采用样本标准差 （n-1）
age_std = df['age'].std(ddof=0)
# 采用总体标准差 （n）
salary_std = df['salary'].std()
print(f"原始数据：{df}")
print(f"age的均值：{age_mean}标准差：{age_std}\nsalary的均值：{salary_mean}标准差：{salary_std}")
# with_mean : 是否对数据居中（减去均值） boolean, 默认为 True；对于稀疏矩阵，False推荐且默认，居中会破坏矩阵的稀疏性。
# with_std :是否将数据缩放到单位方差（除以标准差）； x -μ 只进行居中处理。
standard_scaler = StandardScaler()
df_numeric = df[['age', 'salary']]
df_standardization = standard_scaler.fit_transform(df_numeric)
print(f"标准差后数据：{df_standardization}")
# 采用总体标准差
print(f"特征的均值：{standard_scaler.mean_}方差：{standard_scaler.var_}标准差：{standard_scaler.scale_}\n"
      f"样本数：{standard_scaler.n_samples_seen_}特征数量：{standard_scaler.n_features_in_}特征名称：{standard_scaler.n_features_in_}")
