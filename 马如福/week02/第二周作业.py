# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务：判断5维向量中最大值的维度
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)  # 计算交叉熵损失
        else:
            return torch.softmax(x, dim=1)  # 输出softmax概率分布


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，找出最大值的索引作为标签
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 找到最大值的索引
    return x, max_index


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    # 将模型设置为评估模式
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的样本数
    class_counts = [0] * 5
    for label in y:
        class_counts[label] += 1
    print("各类别样本数量:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，得到概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 取概率最大的类别

        for pred, true in zip(predicted_classes, y):
            if pred == true:
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 错误预测个数：%d, 正确率：%f" % (correct, wrong, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 50  # 增加训练轮数
    batch_size = 32  # 批大小
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 输出类别数（0-4）
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
        # 将模式设置为训练模式
        model.train()
        watch_loss = []

        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据
            start_idx = batch_index * batch_size
            end_idx = (batch_index + 1) * batch_size
            x = train_x[start_idx:end_idx]
            y = train_y[start_idx:end_idx]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(avg_loss)])

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")

    print(log)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.subplot(1, 2, 2)
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
    # 将模式设置为测试模式
    model.eval()
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        probabilities = res.numpy()
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, predicted_class, probabilities))


if __name__ == "__main__":
    main()

    # 测试预测
    test_vec = [
        [0.8, 0.2, 0.8, 0.3, 0.4],
        [0.8, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.7, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.9, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.9]
    ]
    predict("multiclass_model.bin", test_vec)
