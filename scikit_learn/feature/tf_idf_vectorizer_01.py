# TfidfVectorizer：继承于CountVectorizer
# TF (Term Frequency - 词频)： 一个词在当前文档中出现的频率。
# 计算公式： TF(t) = (词t在当前文档中出现的次数) / (当前文档中所有词的总数)
# IDF (Inverse Document Frequency - 逆文档频率)： 一个词在整个语料库中的普遍重要性。
# 计算公式： IDF(t) = log( (总文档数) / (包含词t的文档数 + 1) ) 备注： log底数是10
# TF-IDF： 将两者相乘。
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

data = [
    'A股国产软件概念股全线大涨，开普云、正元智慧、君逸数码等好多强势大涨，另有好多只概念股大涨。'
    '消息面上，全新一代中国操作系统——银河麒麟操作系统在“2025中国操作系统产业大会”正式发布。',
    '上海壹号院五批次好多开盘，数套房源1小时开盘售罄，劲销好多。至此好多，上海壹号院今年好多累'
    '计好多开盘总销售金额超，好多继续保持全国单盘销冠位置。',
    '当日，在江苏南京举行的2025江苏省城市足球联赛好多第九轮比赛中，南京队对阵盐城队。南京市在部分商场、街区'
    '等地设置好多观赛“第二现场”，使用大屏幕同步直播赛事，同时好多设有游戏互动区、烟火市集区，让球迷们度过欢乐时光。'
]

new_corpus = [' '.join(jieba.cut(x)) for x in data]
tf_idf_vectorizer_cn = TfidfVectorizer(min_df=1,
                                       stop_words=[line.strip() for line in
                                                   open('../resource/stopWords.txt', encoding='UTF-8').readlines()])
x_tf_idf_cn = tf_idf_vectorizer_cn.fit_transform(new_corpus)
feature_name = tf_idf_vectorizer_cn.get_feature_names_out()
print(f"中文词汇表（特征名）:\n{feature_name}")
print(f"中文特征矩阵(稀疏矩阵):\n{x_tf_idf_cn.toarray()}")
first_tf_idf_vector = x_tf_idf_cn[0].toarray()[0]
print(f"词汇表和第一行特征矩阵关系:\n{list(zip(feature_name, first_tf_idf_vector))}")
print(f"词汇表和第一行特征矩阵关系带条件:")
"""
sorted_weights = sorted(zip(feature_name, first_tf_idf_vector), key=lambda x: x[1], reverse=True)
for word, weight in sorted_weights:
    if weight > 0.2:
        print(f"{word}: {weight:.4f}")
"""
[print(f"{word}: {weight:.4f}") for word, weight in
 sorted(zip(feature_name, first_tf_idf_vector), key=lambda x: x[1], reverse=True) if weight > 0.2]
