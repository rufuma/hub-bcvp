# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        #self.vocab = load_vocab(config["vocab_path"])
        #self.config["vocab_size"] = len(self.vocab)
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"])
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.max_length = config["max_length"]
        self.ignore_index = config["ignore_index"]
        self.offset_mappings = []
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if not segment.strip():
                    continue
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                raw_sentence = "".join(sentenece)
                self.sentences.append(raw_sentence)
                # 使用BERT Tokenizer编码
                encoder = self.tokenizer(raw_sentence,
                                         return_offsets_mapping=True,
                                         add_special_tokens=True,
                                         max_length=self.max_length,
                                         padding="max_length",
                                         truncation=True)
                input_ids = encoder["input_ids"]
                attention_mask = encoder["attention_mask"]
                offsets_mapping = encoder["offset_mapping"]
                self.offset_mappings.append(offsets_mapping)
                # 对齐label
                token_labels = [self.ignore_index] * self.max_length
                for token_id, (start, end) in enumerate(offsets_mapping):
                    if start == 0 and end == 0:
                        continue
                    char_idx = start
                    if char_idx < len(labels):
                        token_labels[token_id] = labels[char_idx]

                # 转换为tensor,加入数据列表
                #labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids),
                                  torch.LongTensor(attention_mask),
                                  torch.LongTensor(token_labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

