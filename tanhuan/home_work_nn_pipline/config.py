# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/文本分类练习.csv",
    "valid_data_path": "../data/文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"rnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"/Users/tanhuan/Downloads/八斗/week6 语言模型和预训练/bert-base-chinese",
    "seed": 987,
    "train_rate": 0.7,
    # "verify_rate": 0.3,
    "class_num": 2
}

