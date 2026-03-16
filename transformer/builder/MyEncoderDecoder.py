"""
编码器-解码器结构
"""
import torch
from torch import nn
from transformer.encoderLayer.MyEncoder import myEncoder
from transformer.decoderLayer.MyDecoder import myDecoder
from transformer.outputLayer.MyGenerator import myGenerator


class MyEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param src_embed: 源数据嵌入函数
        :param tgt_embed: 目标数据嵌入函数
        :param generator:  输出部分类别生成器对象
        """
        super(MyEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        :param source: 源数据
        :param target: 目标数据
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        :return:
        """
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, src_mask):
        return self.encoder(self.src_embed(source), src_mask)

    def decode(self, memory, src_mask, taget, target_mask):
        # memory:编码器编码后的输出张量
        return self.decoder(self.tgt_embed(taget), memory, src_mask, target_mask)


d_model = 512
vocab_size = 1000
encoder = myEncoder
decoder = myDecoder
src_embed = tgt_embed = nn.Embedding(vocab_size, d_model)
generator = myGenerator

source = target = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
source_mask = target_mask = torch.zeros(2, 4, 4)

myEncoderDecoder = MyEncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
ed_result = myEncoderDecoder(source, target, source_mask, target_mask)

if __name__ == "__main__":
    print(ed_result.shape)
    print(ed_result)
