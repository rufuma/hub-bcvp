# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {
            0: 'O',
            1: 'B-PERSON', 2: 'I-PERSON',
            3: 'B-ORGANIZATION', 4: 'I-ORGANIZATION',
            5: 'B-LOCATION', 6: 'I-LOCATION',
            7: 'B-TIME', 8: 'I-TIME'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        if hasattr(self.config, "model"):
            self.config["model"].num_labels = self.config["class_num"]
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        # 解析附件中的NER数据（格式：字符 O, 字符 O, ...）
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 分割字符和标签对（处理 ", O" 格式）
                pairs = re.split(r',\s*', line)
                chars = []
                labels = []
                for i in range(0, len(pairs)-1, 2):  # 每两个元素为一组（字符, 标签）
                    char = pairs[i].strip()
                    label = pairs[i+1].strip()
                    if char and label:
                        chars.append(char)
                        labels.append(self.label_to_index.get(label, self.label_to_index["O"]))  # 未知标签映射为O
                # 编码文本
                text = ''.join(chars)
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], padding="max_length", truncation=True)
                else:
                    input_id = self.encode_sentence(text)
                # 处理标签：padding和截断
                label_ids = self.padding(labels)
                # 转换为tensor
                input_id = torch.LongTensor(input_id)
                label_ids = torch.LongTensor(label_ids)
                self.data.append([input_id, label_ids])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
