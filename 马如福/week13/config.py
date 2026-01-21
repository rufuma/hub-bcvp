# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train.txt",
    "valid_data_path": "data/test.txt",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 128,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "tuning_tactics": "lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\AI学习\第六周\第六周 语言模型\bert-base-chinese",
    "seed": 987,
    "class_num": 9,
}
