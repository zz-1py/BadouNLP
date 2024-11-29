# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的多分类任务
规律：x是一个5维向量，取最大的数字所在的维度作为类别（0-4）
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，num_classes表示分类数量
        self.activation = torch.softmax  # softmax归一化函数
        self.loss = nn.functional.cross_entropy  # 损失函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y.view(-1).long())  # 计算交叉熵损失
        else:
            return self.activation(x, dim=1)  # 返回预测结果，通过softmax函数归一化


# 生成一个样本，样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 找到最大值的索引作为类别
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 类别保持为标量
    return torch.FloatTensor(X), torch.LongTensor(Y)#标签要为long


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1)  # 取最大类的索引作为预测类别
        correct = (predicted_classes == y).sum().item()  # 计算正确预测的个数
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画损失曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size, 5)  # 类别数量为5
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
   # predicted_class = torch.argmax(result, dim=1)  # 获取预测类别
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
