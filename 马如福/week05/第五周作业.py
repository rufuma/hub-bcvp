#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类  # 程序功能说明：使用预训练的词向量模型
#聚类采用Kmeans算法  # 程序使用的聚类算法：KMeans

import math  # 导入数学运算模块，用于后续计算聚类数量
import re  # 导入正则表达式模块（本程序未实际使用，可能为预留扩展）
import json  # 导入JSON处理模块（本程序未实际使用，可能为预留扩展）
import jieba  # 导入结巴分词库，用于中文文本分词处理
import numpy as np  # 导入numpy库并简写为np，用于数值计算和向量操作
from gensim.models import Word2Vec  # 从gensim库导入Word2Vec类，用于加载词向量模型
from sklearn.cluster import KMeans  # 从sklearn库导入KMeans类，用于执行聚类算法
from collections import defaultdict  # 导入defaultdict，用于按聚类标签分组存储句子


#输入模型文件路径  # 函数功能说明：加载词向量模型
#加载训练好的模型
def load_word2vec_model(path):  # 定义加载词向量模型的函数，参数path为模型文件路径
    model = Word2Vec.load(path)  # 调用Word2Vec的load方法加载指定路径的模型
    return model  # 返回加载成功的词向量模型


def load_sentence(path):  # 定义加载句子的函数，参数path为句子文件路径
    sentences = set()  # 初始化集合用于存储句子（利用集合特性自动去重）
    with open(path, encoding="utf8") as f:  # 以UTF-8编码打开文件，使用with确保文件正确关闭
        for line in f:  # 逐行读取文件内容
            sentence = line.strip()  # 去除每行首尾的空白字符（如换行符、空格）
            # 对句子进行结巴分词，并用空格连接分词结果，添加到集合中
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))  # 打印去重后的句子总数
    return sentences  # 返回处理后的句子集合


#将文本向量化  # 函数功能说明：将句子转换为向量表示
def sentences_to_vectors(sentences, model):  # 定义句子向量化函数，参数为句子集合和词向量模型
    vectors = []  # 初始化列表用于存储所有句子的向量
    for sentence in sentences:  # 遍历每个句子
        words = sentence.split()  # 按空格分割句子（恢复分词结果），得到词语列表
        # 初始化句子向量为全0向量，维度与词向量模型的维度一致
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量（句向量生成方式）
        for word in words:  # 遍历句子中的每个词语
            try:  # 尝试获取词向量（处理未登录词）
                vector += model.wv[word]  # 将当前词的向量累加到句子向量中
            except KeyError:  # 若词语不在词向量模型中（抛出KeyError）
                # 未登录词用全0向量代替，不影响累加结果
                vector += np.zeros(model.vector_size)
        # 将累加的向量除以词数得到平均向量，作为最终句向量，添加到列表
        vectors.append(vector / len(words))
    return np.array(vectors)  # 将列表转换为numpy数组返回（便于后续聚类计算）


def calculate_avg_distance_and_sort(X, labels, centroids):
    """
    :param X: 数据点
    :param labels: 聚类标签
    :param centroids: 聚类中心
    :return: 返回的结果是按照平均值按照从小到大排序的
    """
    # 去重，并返回排序后的结果
    unique_labels = np.unique(labels)
    cluster_distances = {}

    for label in unique_labels:
        # 获取当前簇的所有点
        cluster_points = X[labels == label]
        cluster_center = centroids[label]

        if len(cluster_points) > 0:
            # 计算每个点到簇中心的欧式距离
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            # 计算平均距离
            avg_distance = np.mean(distances)
            cluster_size = len(cluster_points)

            cluster_distances[label] = {
                'avg_distance': avg_distance,
                'size': cluster_size
            }

    return sorted(cluster_distances.items(), key=lambda x: x[1]["avg_distance"])


def main():  # 定义主函数，程序核心逻辑入口
    model = load_word2vec_model(r"model.w2v")  # 调用函数加载名为"model.w2v"的词向量模型
    sentences = load_sentence("titles.txt")  # 调用函数加载"titles.txt"文件中的句子
    # 将所有句子转换为向量表示
    vectors = sentences_to_vectors(sentences, model)

    # 聚类数量设置为句子总数的平方根（经验值，可根据实际需求调整）
    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)  # 打印设置的聚类数量
    kmeans = KMeans(n_clusters)  # 初始化KMeans聚类器，指定聚类数量
    kmeans.fit(vectors)          # 使用句子向量训练聚类器，执行聚类计算
    distance = calculate_avg_distance_and_sort(vectors, kmeans.labels_, kmeans.cluster_centers_)
    for cluster_id, info in distance:
        print(f"簇 {cluster_id}: 平均距离 = {info['avg_distance']:.4f}, 包含点数 = {info['size']}")


# 当脚本被直接运行时（而非被导入为模块），执行main函数
if __name__ == "__main__":
    main()
