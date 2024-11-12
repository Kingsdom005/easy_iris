import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import iris_dataloader


# 初始化神经网络模型

class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# 定义计算环境
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练集、验证集和测试集
custom_dataset = iris_dataloader("./iris_data/iris.data")
train_size = int(len(custom_dataset) * 0.7)
validate_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - train_size - validate_size

train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,
                                                                              [train_size, validate_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # batch_size=每次16个数据
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"训练集的大小：{len(train_loader) * 16}, 验证集的大小：{len(train_loader)}，测试集的大小：{len(train_loader)}")


# 定义推理函数，来验证并返回准确率
def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():  # 不改变模型参数
        for data in dataset:
            datas, label = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # ?
            acc_num += torch.eq(predict_y, label.to(device)).sum().item()

    acc = acc_num / len(dataset)
    return acc


def main(lr=0.005, epoches=20):
    model = NN(4, 12, 6, 3).to(device)  # 12 6 hyper para

    loss_f = nn.CrossEntropyLoss()

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(pg, lr=lr)

    # 权重文件存储路径
    save_path = os.path.join(os.getcwd(), "results/weights")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 开始训练
    for epoch in range(epoches):
        model.train()
        acc_num = torch.zeros(1).to(device)  # ?
        sample_num = 0

        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)

        for datas in train_bar:
            data, label = datas
            label = label.squeeze(-1)  # ?
            sample_num += data.shape[0]

            optimizer.zero_grad()  # 清空上次结果

            outputs = model(data.to(device))

            pred_class = torch.max(outputs, dim=1)[1]  # torch.max返回一个元组，第一个元素是max的值，第二元素是max的索引
            acc_num = torch.eq(pred_class, label.to(device)).sum()

            loss = loss_f(outputs, label.to(device).long())
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epoches, loss)

        val_acc = infer(model, validate_loader, device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc:{:.3f} validate_acc:{:.3f}".format(epoch + 1, epoches, loss,
                                                                                           train_acc, val_acc))

        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        # 每次迭代之后，对初始化指标清零
        train_acc = 0.
        val_acc = 0.
    print("Finished Training.")

    test_acc = infer(model, test_loader, device)
    print("test_gcc:", test_acc)


if __name__ == "__main__":
    main()
