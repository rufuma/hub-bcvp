# coding:utf8

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
多分类任务：x是一个5维向量，最大值所在的维度即为类别（0-4）

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出改为5个节点，对应5个类别
        # 移除sigmoid激活，因为CrossEntropyLoss内部包含Softmax
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 使用交叉熵损失
        else:
            return x  # 直接输出logits


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值所在的维度即为类别（0-4）
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 最大值所在的维度索引
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # y现在是0-4的整数，不是列表
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意Y是LongTensor


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计各类别数量
    class_counts = [0] * 5
    for label in y:
        class_counts[label] += 1
    print("各类别样本数量:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 获取预测类别
        for y_p, y_t in zip(predicted, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 输出类别数
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        probabilities = torch.softmax(result, dim=1)  # 转换为概率
        _, predicted = torch.max(result, 1)  # 获取预测类别

    for vec, pred, prob in zip(input_vec, predicted, probabilities):
        actual_class = np.argmax(vec)
        print("输入：%s, 实际类别：%d, 预测类别：%d, 概率值：%s" %
              (vec, actual_class, pred.item(), [f"{p:.4f}" for p in prob.numpy()]))


if __name__ == "__main__":
    main()

    # 测试预测
    # test_vec = [
    #     [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 最大值在索引4
    #     [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],  # 最大值在索引2
    #     [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # 最大值在索引0
    #     [0.19349776, 0.89416669, 0.22579291, 0.41567412, 0.1358894]  # 最大值在索引1
    # ]
    # predict("multiclass_model.bin", test_vec)
