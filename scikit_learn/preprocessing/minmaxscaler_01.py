import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'age': [25, 30, 34, 45, 60, 30, 15],
    'salary': [50000, 54000, 60000, 54000, 100000, 40000, 20000],
    'country': ['USA', 'UK', 'China', 'USA', 'India', 'China', 'UK'],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'F']
}
df = pd.DataFrame(data)
print(f"原始数据：{df}")
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
df_numeric = df[['age', 'salary']]
df_normalization = minmax_scaler.fit_transform(df_numeric)
print(f"归一化后数据：{df_normalization}")
