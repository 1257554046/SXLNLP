# -*- coding: utf-8 -*-
import csv
import time

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, 1)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    cost_time = []
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc, predict_100_cost_time = evaluator.eval(epoch)
        cost_time.append(predict_100_cost_time)
    # model_name = "{}_epoch_{}.pth".format(Config["model_type"], epoch)
    # model_path = os.path.join(config["model_path"], model_name)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    # 获取预测100条平均时间
    predict_100_average_cost_time = np.mean(cost_time)
    return {
        "model_type": config["model_type"],
        "learning_rate": config["learning_rate"],
        "hidden_size": config["hidden_size"],
        "batch_size": config["batch_size"],
        "pooling_style": config["pooling_style"],
        "acc": round(acc, 4) * 100,
        "predict_100_average_cost_time": round(float(predict_100_average_cost_time), 4) * 1000,
    }


def write_to_csv(data_to_append):
    # 定义文件名
    filename = 'data.csv'
    # 定义表头
    headers = ['Model', 'Learning_Rate', 'Hidden_Size', 'Batch_Size', 'Pooling_Style', 'Acc(%)',
               'Predict_100_average_cost_time(ms)']
    # 定义要追加的数据（每一行是一个列表）
    # data_to_append = [
    #     ['value1_1', 'value1_2', 'value1_3', 'value1_4'],
    #     ['value2_1', 'value2_2', 'value2_3', 'value2_4']
    # ]

    # 检查文件是否存在
    file_exists = os.path.isfile(filename)

    # 打开文件，若文件不存在则创建
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(headers)

        # 写入数据
        writer.writerows(data_to_append)


if __name__ == "__main__":
    # main(Config)
    append_data = []
    # for model in ["bert"]:
    #     Config["model_type"] = model
    #     # time_start = time.time()
    #     # print("最后一轮准确率：", main(Config)*100, "当前配置：", Config["model_type"])
    #     result = main(Config)
    #     print(result)
    #     append_data.append([i for i in result.values()])
    #     # print("最后一轮准确率：%.4f%%, 当前配置：%s" % (main(Config)*100, Config["model_type"]))
    #     # print("%s 耗时：%f" % (Config["model_type"], time.time()-time_start))

    for model in ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn",
                  "rcnn", "bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        result = main(Config)
                        print("最后一轮准确率：", result, "当前配置：", Config)
                        print(result)
                        append_data.append([i for i in result.values()])
    write_to_csv(append_data)

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg"]:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)
