import math
import torch
from torch import nn
from torch.nn import functional as F
import re
from collections import Counter
from datasets import load_dataset
from pyitcast.transformer import TransformerModel
import time
import copy


class Vocab:
    """词汇表类，模拟 torchtext 的 Vocab"""

    def __init__(self, stoi_dict, itos_dict):
        self.stoi = stoi_dict  # string to index
        self.itos = itos_dict  # index to string

    def __len__(self):
        return len(self.stoi)


class TextTokenizer:
    """纯 PyTorch 实现的文本分词器，替代 torchtext"""

    def __init__(self, init_token="<sos>", eos_token="<eos>",
                 pad_token="<pad>", unk_token="<unk>"):
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        # 特殊标记
        self.specials = [pad_token, unk_token, init_token, eos_token]
        self.vocab_dict = {token: idx for idx, token in enumerate(self.specials)}
        self.idx2word = {idx: token for token, idx in self.vocab_dict.items()}
        self.vocab = None  # 将在 build_vocab 后创建

    def tokenize(self, text):
        """基础英文分词，模拟 torchtext 的 basic_english"""
        text = text.lower().strip()
        # 在标点符号前后添加空格，然后分割
        text = re.sub(r'([.,!?;:"()\[\]{}])', r' \1 ', text)
        tokens = text.split()
        return tokens

    def build_vocab(self, texts, min_freq=1):
        """从文本列表构建词汇表
        Args:
            texts: 文本列表，可以是字符串列表或 Example 对象列表
            min_freq: 最小词频
        """
        counter = Counter()

        for item in texts:
            # 处理不同类型的输入：字符串或 Example 对象
            if isinstance(item, str):
                text = item
            elif hasattr(item, 'text'):
                # Example 对象，有 text 属性
                text = item.text
                if isinstance(text, list):
                    # text 已经是 token 列表
                    counter.update(text)
                    continue
            else:
                text = str(item)

            counter.update(self.tokenize(text))

        # 添加频率足够的词
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.vocab_dict:
                idx = len(self.vocab_dict)
                self.vocab_dict[word] = idx
                self.idx2word[idx] = word

        # 创建 vocab 对象，兼容 torchtext API
        self.vocab = Vocab(self.vocab_dict.copy(), self.idx2word.copy())

        return self

    def numericalize(self, texts):
        """将文本列表转换为数字索引张量
        Args:
            texts: 文本列表，每个元素可以是字符串或 token 列表
        Returns:
            torch.Tensor: 索引张量
        """
        # 当texts是[[token1, token2, ...]]格式时（单个长文本）
        if len(texts) == 1 and isinstance(texts[0], list):
            tokens = texts[0]
            # 转换为索引（不添加额外特殊标记，因为数据已包含）
            indices = [self.vocab_dict.get(token, self.vocab_dict[self.unk_token]) for token in tokens]
            return torch.tensor(indices, dtype=torch.long).view(-1, 1)

        # 处理多个短文本的情况
        indices_list = []
        for item in texts:
            if isinstance(item, str):
                tokens = self.tokenize(item)
            elif isinstance(item, list):
                tokens = item
            else:
                tokens = [str(item)]

            # 添加特殊标记
            tokens = [self.init_token] + tokens + [self.eos_token]

            # 转换为索引
            indices = [self.vocab_dict.get(token, self.vocab_dict[self.unk_token]) for token in tokens]
            indices_list.append(indices)

        # 找到最大长度进行填充
        max_len = max(len(indices) for indices in indices_list)
        padded_indices = []

        for indices in indices_list:
            # 用 pad_token 的索引填充
            padded = indices + [self.vocab_dict[self.pad_token]] * (max_len - len(indices))
            padded_indices.append(padded)

        return torch.tensor(padded_indices, dtype=torch.long)

    def encode(self, text):
        """将文本编码为索引序列"""
        tokens = [self.init_token] + self.tokenize(text) + [self.eos_token]
        return [self.vocab_dict.get(token, self.vocab_dict[self.unk_token]) for token in tokens]

    def decode(self, indices):
        """将索引序列解码为文本"""
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        return " ".join(words)

    def __len__(self):
        return len(self.vocab_dict)


# 第一步：加载 WikiText-2 数据集
# 创建语料域, 语料域是存放语料的数据结构,
# 它的四个参数代表给存放语料（或称作文本）施加的作用.
# 分别为 tokenize,使用get_tokenizer("basic_english")获得一个分割器对象,
# 分割方式按照文本为基础英文进行分割.
# init_token为给文本施加的起始符 <sos>给文本施加的终止符<eos>,
# 最后一个lower为True, 存放的文本字母全部小写.
TEXT = TextTokenizer(init_token="<sos>", eos_token="<eos>")

# 最终获得一个Field对象.
# <torchtext.data.field.Field object at 0x7fc42a02e7f0>

# 然后使用datasets库导入WikiText2数据,
# 并切分为对应训练文本, 验证文本，测试文本, 并对这些文本施加刚刚创建的语料域.
from datasets import load_dataset

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 过滤空文本并合并成一个长文本序列（与torchtext的WikiText2格式一致）
train_tokens = []
for text in raw_dataset["train"]["text"]:
    if text.strip():
        train_tokens.extend(TEXT.tokenize(text))
        train_tokens.append("<eos>")  # 每个段落结束添加eos

val_tokens = []
for text in raw_dataset["validation"]["text"]:
    if text.strip():
        val_tokens.extend(TEXT.tokenize(text))
        val_tokens.append("<eos>")

test_tokens = []
for text in raw_dataset["test"]["text"]:
    if text.strip():
        test_tokens.extend(TEXT.tokenize(text))
        test_tokens.append("<eos>")


# 创建简单的Example类来兼容demo.py的接口
class Example:
    def __init__(self, text):
        self.text = text


class Dataset:
    def __init__(self, tokens):
        self.examples = [Example(tokens)]


train_txt = Dataset(train_tokens)
val_txt = Dataset(val_tokens)
test_txt = Dataset(test_tokens)

# 将训练集文本数据构建一个vocab对象,
# 这样可以使用vocab对象的stoi方法统计文本共包含的不重复词汇总数.
TEXT.build_vocab(train_tokens, min_freq=1)

# 然后选择设备cuda或者cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 第二步：构建用于模型输入的批次化数据
def batchify(data, batch_size):
    """batchify函数用于将文本数据映射成连续数字, 并转换成指定的样式, 指定的样式可参考下图.
       它有两个输入参数, data就是我们之前得到的文本数据(train_txt, val_txt, test_txt),
       batch_size是就是batch_size, 每次模型更新参数的数据量"""
    # 使用TEXT的numericalize方法将单词映射成对应的连续数字.
    data = TEXT.numericalize([data.examples[0].text])

    # 接着用数据词汇总数除以batch_size,
    # 取整数得到一个nbatch代表需要多少次batch后能够遍历完所有数据
    nbatch = data.size(0) // batch_size

    # 之后使用narrow方法对不规整的剩余数据进行删除,
    # 第一个参数是代表横轴删除还是纵轴删除, 0为横轴，1为纵轴
    # 第二个和第三个参数代表保留开始轴到结束轴的数值.类似于切片
    # 可参考下方演示示例进行更深理解.
    data = data.narrow(0, 0, nbatch * batch_size)
    # 因为会做转置操作, 因此这个矩阵的形状是[None, batch_size],
    # 如果输入是训练数据的话，形状为[104335, 20], 可以通过打印data.shape获得.
    # 也就是data的列数是等于batch_size的值的.
    data = data.view(batch_size, -1).t().contiguous()
    # 最后将数据分配在指定的设备上.
    return data.to(device)


batch_size = 20
eval_batch_size = 10

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# 设置训练批次数据大小
bptt = 35  # 令子长度允许的最大值bptt为35（超参数）


def get_batch(source, i):
    """用于获得每个批次合理大小的源数据和目标数据.
       参数source是通过batchify得到的train_data/val_data/test_data.
       i是具体的批次次数.
    """

    # 首先我们确定句子长度, 它将是在bptt和len(source) - 1 - i中最小值
    # 实质上, 前面的批次中都会是bptt的值, 只不过最后一个批次中, 句子长度
    # 可能不够bptt的35个, 因此会变为len(source) - 1 - i的值.
    seq_len = min(bptt, len(source) - 1 - i)

    # 语言模型训练的源数据的第i批数据将是batchify的结果的切片[i:i+seq_len]
    data = source[i:i + seq_len]

    # 根据语言模型训练的语料规定, 它的目标数据是源数据向后移动一位
    # 因为最后目标数据的切片会越界, 因此使用view(-1)来保证形状正常.
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


# 第三步：构建训练和评估函数
# 设置训练超参数
# 获得不重复词汇总数
ntokens = len(TEXT.vocab)
# 词嵌维度
emsize = 200
# 前馈全连接层节点数
nhid = 200
# 编码器层数
nlayers = 2
# 多头注意力机制头数
nhead = 2
# 置零比率
dropout = 0.2
# 将参数输入到TransformerModel中
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
# 关于损失函数, 我们使用nn自带的交叉熵损失
entropy_loss = nn.CrossEntropyLoss()
# 学习速率
lr = 0.5
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.1, gamma=0.95)


def train():
    # 模型开启训练
    model.train()
    # 定义初始数据
    total_loss = 0
    start_time = time.time()
    # 遍历获取批次数据
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 获取源数据和目标数据
        data, targets = get_batch(train_data, i)
        # 初始化优化器梯度为0
        optimizer.zero_grad()
        # 模型训练获得结果
        output = model(data)
        # 将输出和目标数据传入损失函数对象
        loss = entropy_loss(output.view(-1, ntokens), targets)
        # 反向传播
        loss.backward()
        # 梯度规范化, 防止出现梯度消失或爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 模型参数更新
        optimizer.step()
        # 计算总损失
        total_loss += loss.item()
        # 日志打印间隔
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            # 平均损失
            cur_loss = total_loss / log_interval
            # 时间消耗
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt,
                scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            # 每个批次结束后, 总损失归0
            total_loss = 0
            # 开始时间取当前时间
            start_time = time.time()


def evaluate(eval_model, data_source):
    """
    :param eval_model: 轮训练产生的模型
    :param data_source: 验证或测试数据集
    """
    # 模型评估
    eval_model.eval()
    # 初始总损失值
    total_loss = 0
    # 因为评估模式模型参数不变, 因此反向传播不需要求导, 以加快计算
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            # 对输出形状扁平化, 变为全部词汇的概率分布
            output_flat = output.view(-1, ntokens)
            # 评估过程总损失值
            total_loss += entropy_loss(output_flat, targets).item()
            # 平均损失
            cur_loss = total_loss / ((data_source.size(0) - 1) / bptt)

    return cur_loss


# 第四步：进行训练和评估(包括验证以及测试)
# 首先初始化最佳验证损失，初始值为无穷大
best_val_loss = float("inf")
# 定义最佳模型变量
best_model = None
# 轮训次数
epochs = 3
for epoch in range(1, epochs + 1):
    # 首先获得轮数开始时间
    epoch_start_time = time.time()
    # 训练模型
    train()
    # 评估模型
    val_loss = evaluate(model, val_data)
    print('-' * 60)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 60)
    # 根据损失函数获取最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
    # 每轮都会对优化方法的学习率做调整
    scheduler.step()

# 模型测试代码
test_loss = evaluate(best_model, test_data)
print('=' * 60)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 60)
