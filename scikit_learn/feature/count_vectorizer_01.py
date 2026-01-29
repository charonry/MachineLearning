# 词袋模型 - CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba

# 1.英文
# 示例文本数据
corpus = [
    'I love love machine learning',
    'Machine learning is fun',
    'I love coding in Python'
]
# min_df: 忽略文档频率低于此值的词
# stop_words（默认None）: 移除停用词（如 'the', 'is', 'and'）
count_vectorizer = CountVectorizer(min_df=1, stop_words='english')
# 学习词汇字典并转换文本数据
x_count = count_vectorizer.fit_transform(corpus)
# 查看生成的词汇表
print(f"词汇表（特征名）:\n{count_vectorizer.get_feature_names_out()}")
print(f"特征矩阵:\n{x_count}")
print(f"特征矩阵(稀疏矩阵):\n{x_count.toarray()}")

# 2.中文
data = [
    'A股国产软件概念股全线大涨，开普云、正元智慧、君逸数码等好多强势大涨，另有好多只概念股大涨。'
    '消息面上，全新一代中国操作系统——银河麒麟操作系统在“2025中国操作系统产业大会”正式发布。',
    '上海壹号院五批次好多开盘，数套房源1小时开盘售罄，劲销好多。至此好多，上海壹号院今年好多累'
    '计好多开盘总销售金额超，好多继续保持全国单盘销冠位置。',
    '当日，在江苏南京举行的2025江苏省城市足球联赛好多第九轮比赛中，南京队对阵盐城队。南京市在部分商场、街区'
    '等地设置好多观赛“第二现场”，使用大屏幕同步直播赛事，同时好多设有游戏互动区、烟火市集区，让球迷们度过欢乐时光。'
]

new_corpus = [' '.join(jieba.cut(x)) for x in data]
count_vectorizer_cn = CountVectorizer(min_df=1,
                                      stop_words=[line.strip() for line in
                                                  open('../resource/stopWords.txt', encoding='UTF-8').readlines()])
x_count_cn = count_vectorizer_cn.fit_transform(new_corpus)
print(f"中文词汇表（特征名）:\n{count_vectorizer_cn.get_feature_names_out()}")
print(f"中文特征矩阵(稀疏矩阵):\n{x_count_cn.toarray()}")
