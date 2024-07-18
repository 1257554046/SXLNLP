#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

from collections import OrderedDict

class OrderedSet(OrderedDict):
    def add(self, key):
        OrderedDict.setdefault(self, key, None)

    def __init__(self, init_list=None):
        super().__init__()
        if init_list is not None:
            for item in init_list:
                self.add(item)
#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    # 使用有序集合，便于保存文本序号信息
    sentences = OrderedSet()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model, tfidf_dict):
    vectors = []
    for index, sentence in enumerate(sentences):
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                # 获取词对应的tfidf
                word_tfidf = tfidf_dict[index].get(word, 0)
                # 作为重要性权重相乘
                vector += (model.wv[word] * word_tfidf)
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)
def cosine_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))


#统计tf和idf值
def build_tf_idf_dict(corpus):
    tf_dict = defaultdict(dict)  #key:文档序号，value：dict，文档中每个词出现的频率
    idf_dict = defaultdict(set)  #key:词， value：set，文档序号，最终用于计算每个词在多少篇文档中出现过
    for text_index, text_words in enumerate(corpus):
        for word in text_words:
            if word not in tf_dict[text_index]:
                tf_dict[text_index][word] = 0
            tf_dict[text_index][word] += 1
            idf_dict[word].add(text_index)
    idf_dict = dict([(key, len(value)) for key, value in idf_dict.items()])
    return tf_dict, idf_dict

#根据tf值和idf值计算tfidf
def calculate_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = defaultdict(dict)
    for text_index, word_tf_count_dict in tf_dict.items():
        for word, tf_count in word_tf_count_dict.items():
            tf = tf_count / sum(word_tf_count_dict.values())
            #tf-idf = tf * log(D/(idf + 1))
            tf_idf_dict[text_index][word] = tf * math.log(len(tf_dict)/(idf_dict[word]+1))
    return tf_idf_dict

#输入语料 list of string
#["xxxxxxxxx", "xxxxxxxxxxxxxxxx", "xxxxxxxx"]
def calculate_tfidf(corpus):
    #先进行分词
    corpus = [jieba.lcut(text) for text in corpus]
    tf_dict, idf_dict = build_tf_idf_dict(corpus)
    tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
    return tf_idf_dict

def get_tfidf_dict():
    file_path = r"titles.txt"
    # 使用有序集合
    corpus = OrderedSet()
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            corpus.add(line.strip())
    tf_idf_dict = calculate_tfidf(corpus)
    return tf_idf_dict

def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    # 获取每个不重复文本对应的分词后的tfidf
    tf_idf_dict = get_tfidf_dict()
    vectors = sentences_to_vectors(sentences, model, tf_idf_dict)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    # 获取类别质心向量Array
    centers = kmeans.cluster_centers_
    sentence_label_dict = defaultdict(list)
    sentence_distance_dict = defaultdict(float)
    i = 0
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        # 获取文本向量
        sentence_vector = vectors[i]
        # 获取类别质心向量
        center_vector = centers[label]
        # 计算质心和文本向量距离
        distance = cosine_distance(sentence_vector, center_vector)
        sentence_distance_dict[label] += distance
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        i += 1
    # 计算每一类平均距离
    for label, sum_distance in sentence_distance_dict.items():
        sentence_count = len(sentence_label_dict[label])
        sentence_distance_dict[label] = sum_distance / sentence_count
    # 根据平均距离进行排序
    sorted_list = sorted([(label, average_dis) for label, average_dis in sentence_distance_dict.items()], key=lambda x: x[1])

    for label, average_dis in sorted_list[:10]:
        sentences = sentence_label_dict[label]
        print("cluster %s :" % label)
        print("average_dis %s" % average_dis)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

