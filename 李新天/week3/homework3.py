#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
“你”出现在字符串中第几个位置，就是第几类
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)    #RNN层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.Linear(hidden_size, sentence_length)     #线性层
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.rnn(x)[0]                         #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, hidden_size) -> (batch_size, hidden_size, sen_len)
        x = self.pool(x)                           #(batch_size, hidden_size, sen_len)->(batch_size, hidden_size, 1)
        x = x.squeeze()                            #(batch_size, hidden_size, 1) -> (batch_size, hidden_size)
        y_pred = self.classify(x)                  #(batch_size, hidden_size) -> (batch_size, sentence_length)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz你我他"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    vocab_keys = list(vocab.keys())
    vocab_keys.remove('你')
    x = [random.choice(vocab_keys) for _ in range(sentence_length - 1)]
    x.append('你')
    random.shuffle(x)
    y = x.index('你')
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y


#建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


#建立模型
def build_model(vocab, char_dim, hidden_size, sentence_length):
    model = TorchModel(char_dim, hidden_size, sentence_length, vocab)
    return model


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    hidden_size = char_dim + 1
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, hidden_size, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


#使用训练好的模型做预测
def predict(model_path, input_strings):
    char_dim = 20  # 每个字的维度
    hidden_size = char_dim + 1
    sentence_length = 6  # 样本文本长度
    vocab = vocab = build_vocab() #加载字符表
    model = build_model(vocab, char_dim, hidden_size, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():
        y_pred = model(torch.LongTensor(x))
    for input_string, y_p in zip(input_strings, y_pred):
        print("输入：%s, 预测类别：%d" % (input_string, torch.argmax(y_p)))


if __name__ == "__main__":
    main()
    test_strings = ["fnss你e", "wsaa你g", "rq我w你qg", "他ssq我你", "你iade"]
    predict("model.pth", test_strings)
