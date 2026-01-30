import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    'age': [25, 30, np.nan, 45, 60, 30, 15],
    'salary': [50000, 54000, 60000, np.nan, 100000, 40000, 20000],
    'country': ['USA', 'UK', 'China', 'USA', 'India', 'China', 'UK'],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'F']
}
df = pd.DataFrame(data)
print(f"原始数据：{df}")
print(f"年龄的平均值：{df['age'].mean()}")
print(f"薪资的平均值：{df['salary'].mean()}")
df['gender'] = df['gender'].astype('category')
df['country'] = df['country'].astype('category')
imputer = SimpleImputer(strategy='mean')
df_numeric = df[['age', 'salary']]
df[['age', 'salary']] = imputer.fit_transform(df_numeric)
print(f"缺失值处理后数据：{df}")
