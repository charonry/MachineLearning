# MachineLearning 机器学习项目

本项目是一个机器学习学习与实践代码库，包含两个主要部分：**scikit-learn 经典机器学习算法** 和 **Transformer 深度学习模型**。

## 项目结构

```
MachineLearning/
├── scikit_learn/          # 基于 scikit-learn 的传统机器学习算法
│   ├── classification/    # 分类算法
│   ├── cluster/           # 聚类算法
│   ├── dataset/           # 数据集加载示例
│   ├── feature/           # 特征工程
│   ├── loadsave/          # 模型保存与加载
│   ├── preprocessing/     # 数据预处理
│   ├── reduce/            # 降维算法
│   └── regression/        # 回归算法
├── transformer/           # Transformer 深度学习模型
│   ├── builder/           # 模型构建器
│   ├── cases/             # 应用案例
│   ├── decoderLayer/      # 解码器层
│   ├── encoderLayer/      # 编码器层
│   ├── inputLayer/        # 输入层
│   ├── operation/         # 模型操作
│   └── outputLayer/       # 输出层
└── demo.py                # 环境检测脚本
```

## 一、scikit-learn 模块

本模块涵盖了机器学习中最常用的算法和工具，每个文件都是独立的可运行示例。

### 1. 分类算法 (classification/)

| 文件                        | 算法         | 说明             |
|---------------------------|------------|----------------|
| `MyKNN.py`                | K-近邻 (KNN) | 基于实例的监督学习分类算法  |
| `MyLogisticRegression.py` | 逻辑回归       | 用于二分类/多分类的线性模型 |
| `MyDecisionTree.py`       | 决策树        | 基于树结构的分类算法     |
| `MyMultinomialNB.py`      | 朴素贝叶斯      | 基于概率的分类算法      |
| `MySGDClassifier.py`      | 随机梯度下降分类器  | 大规模数据集的线性分类器   |

### 2. 回归算法 (regression/)

| 文件                      | 算法      | 说明           |
|-------------------------|---------|--------------|
| `MyLinearRegression.py` | 线性回归    | 最基本的回归算法     |
| `MyRidge.py`            | 岭回归     | 带L2正则化的线性回归  |
| `MyLasso.py`            | Lasso回归 | 带L1正则化的线性回归  |
| `MyElasticNet.py`       | 弹性网络    | L1和L2混合正则化回归 |
| `MySGDRegressor.py`     | SGD回归器  | 随机梯度下降回归     |

### 3. 聚类算法 (cluster/)

| 文件                          | 算法      | 说明          |
|-----------------------------|---------|-------------|
| `MyKMeans.py`               | K-Means | 基于距离的划分聚类算法 |
| `MyDBSCAN.py`               | DBSCAN  | 基于密度的聚类算法   |
| `MyAgglomerativeCluster.py` | 层次聚类    | 自底向上的聚合聚类   |

### 4. 降维算法 (reduce/)

| 文件                       | 算法     | 说明                |
|--------------------------|--------|-------------------|
| `MyPCA.py`               | 主成分分析  | 线性降维算法            |
| `MyLDA.py`               | 线性判别分析 | 有监督降维算法           |
| `MySelectKBest.py`       | 特征选择   | 基于统计检验选择K个最佳特征    |
| `MyVarianceThreshold.py` | 方差阈值   | 基于方差的特征筛选         |
| `MyRandomForest.py`      | 随机森林   | 集成学习算法（也可用于特征重要性） |

### 5. 数据预处理 (preprocessing/)

| 文件                     | 功能    | 说明              |
|------------------------|-------|-----------------|
| `standardscaler_01.py` | 标准化   | 将特征缩放到均值为0，方差为1 |
| `minmaxscaler_01.py`   | 归一化   | 将特征缩放到[0,1]范围   |
| `simpleImputer_01.py`  | 缺失值处理 | 填充数据中的缺失值       |

### 6. 特征工程 (feature/)

| 文件                        | 功能        | 说明                |
|---------------------------|-----------|-------------------|
| `count_vectorizer_01.py`  | 词频向量化     | 将文本转换为词频矩阵        |
| `tf_idf_vectorizer_01.py` | TF-IDF向量化 | 考虑词频和逆文档频率的文本特征提取 |
| `one_hot_encoder_01.py`   | 独热编码      | 将分类变量转换为二进制向量     |
| `dict_vectorizer_01.py`   | 字典向量化     | 将字典列表转换为特征矩阵      |

### 7. 模型持久化 (loadsave/)

| 文件               | 功能   | 说明                 |
|------------------|------|--------------------|
| `MySaveModel.py` | 保存模型 | 使用 joblib 保存训练好的模型 |
| `MyLoadModel.py` | 加载模型 | 加载已保存的模型进行预测       |

---

## 二、Transformer 模块

本模块实现了完整的 Transformer 架构，包含从底层组件到完整语言模型的实现。

### 核心组件

#### 输入层 (inputLayer/)

- `MyInputEmbedding.py` - 词嵌入层
- `MyPositionalEncoding.py` - 位置编码层

#### 编码器层 (encoderLayer/)

- `MyEncoder.py` - 完整编码器
- `MyEncoderLayer.py` - 单层编码器
- `MyAttention.py` - 自注意力机制
- `MyMultiHeadedAttention.py` - 多头注意力机制
- `MyPositionwiseFeedForward.py` - 前馈神经网络
- `MyLayerNorm.py` - 层归一化
- `MySublayerConnection.py` - 子层连接（残差连接）
- `MyMaskTensor.py` - 掩码张量处理

#### 解码器层 (decoderLayer/)

- `MyDecoder.py` - 完整解码器
- `MyDecoderLayer.py` - 单层解码器

#### 输出层 (outputLayer/)

- `MyGenerator.py` - 输出生成器

#### 构建器 (builder/)

- `MyEncoderDecoder.py` - 编码器-解码器架构构建
- `MyMakeModel.py` - 模型工厂类

#### 操作工具 (operation/)

- `DataGenerator.py` - 数据生成器
- `ModelHandler.py` - 模型处理器
- `ModelRuner.py` - 模型运行器
- `ModelRunerGreedyDecode.py` - 贪婪解码运行器

### 应用案例 (cases/)

#### 语言模型 (cases/language/)

- `TextHandler.py` - 基于 WikiText-2 数据集的语言模型训练
    - 使用纯 PyTorch 实现的文本分词器
    - 基于 `pyitcast.transformer` 的 TransformerModel
    - 包含完整的训练、验证、测试流程

---

## 环境要求

### Python 版本

- Python 3.x

### 主要依赖

```
# scikit-learn 模块
numpy
scikit-learn

# transformer 模块
torch
datasets
pyitcast
```

### 安装依赖

```bash
pip install numpy scikit-learn torch datasets pyitcast
```

---

## 使用方法

### 运行 scikit-learn 示例

```bash
# 运行 KNN 分类示例
python scikit_learn/classification/MyKNN.py

# 运行线性回归示例
python scikit_learn/regression/MyLinearRegression.py

# 运行 K-Means 聚类示例
python scikit_learn/cluster/MyKMeans.py
```

### 运行 Transformer 语言模型

```bash
# 运行 WikiText-2 语言模型训练
python transformer/cases/language/TextHandler.py
```

