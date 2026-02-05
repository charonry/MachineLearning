"""
MultinomialNB:朴素贝叶斯 监督学习算法，
分类任务，尤其在文本分类场景
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1.加载数据
news = fetch_20newsgroups(subset='all')
# 2.数据预处理
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)
# 3.特征抽取 TF-IDF
tfidfVectorizer = TfidfVectorizer()
X_train_scaled = tfidfVectorizer.fit_transform(X_train)
X_test_scaled = tfidfVectorizer.transform(X_test)
# 4.创建和训练MultinomialNB模型
multinomialNB = MultinomialNB()  # MultinomialNB分类器
multinomialNB.fit(X_train_scaled, y_train)  # 使用训练集拟合
# 5.进行预测并评估模型
y_pred = multinomialNB.predict(X_test_scaled)
print(f"正确值：{y_test}\n朴素贝叶斯预测值：{y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率：{accuracy:.4f}')
print('分类报告：\n', classification_report(y_test, y_pred, target_names=news.target_names))
