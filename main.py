import numpy as np
import torch
from PIL import Image
import pickle
import os
import torchvision
from torch import nn
from d2l import torch as d2l

def sort_max(row):
   return max(row)

def get_sorted_array(array):
    temp = list(array)
    temp.sort(key=sort_max, reverse=True)
    for t in range(len(temp) - 1, -1, -1):
        if np.sum(temp[t]) > 0:
            break
    return np.array(temp[:t + 1])

def save2image(list_positive, list_negative):
    ROOT_DIR = os.getcwd()

    # 训练集路径
    train_dir = os.path.join(ROOT_DIR, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_positive_dir = os.path.join(train_dir, "positive")
    train_negative_dir = os.path.join(train_dir, "negative")
    if not os.path.exists(train_negative_dir):
        os.mkdir(train_negative_dir)
    if not os.path.exists(train_positive_dir):
        os.mkdir(train_positive_dir)

    # 测试集路径
    test_dir = os.path.join(ROOT_DIR, "test")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    test_positive_dir = os.path.join(test_dir, "positive")
    test_negative_dir = os.path.join(test_dir, "negative")
    if not os.path.exists(test_negative_dir):
        os.mkdir(test_negative_dir)
    if not os.path.exists(test_positive_dir):
        os.mkdir(test_positive_dir)

    # 训练集：测试集 = 2:1
    # 构造训练集
    train_num = len(list_positive)//3 * 2
    for i in range(train_num):
        for j in range(len(list_positive[i])):

            list_positive[i][j] = get_sorted_array(list_positive[i][j])
            min = np.min(list_positive[i][j])
            max = np.max(list_positive[i][j])
            img_array = (list_positive[i][j] - min)/(max - min) * 255
            img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
            img.save(train_positive_dir + "//" + str(i) + "_" + str(j) + ".png")

            list_negative[i][j] = get_sorted_array(list_negative[i][j])
            min = np.min(list_negative[i][j])
            max = np.max(list_negative[i][j])
            img_array = (list_negative[i][j] - min) / (max - min) * 255
            img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
            img.save(train_negative_dir + "//" + str(i) + "_" + str(j) + ".png")
    # 构造测试集
    for i in range(train_num, len(list_positive)):
        for j in range(len(list_positive[i])):
            list_positive[i][j] = get_sorted_array(list_positive[i][j])
            min = np.min(list_positive[i][j])
            max = np.max(list_positive[i][j])
            img_array = (list_positive[i][j] - min)/(max - min) * 255
            img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
            img.save(test_positive_dir + "//" + str(i) + "_" + str(j) + ".png")

            list_negative[i][j] = get_sorted_array(list_negative[i][j])
            min = np.min(list_negative[i][j])
            max = np.max(list_negative[i][j])
            img_array = (list_negative[i][j] - min) / (max - min) * 255
            img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
            img.save(test_negative_dir + "//" + str(i) + "_" + str(j) + ".png")

def save2image_new(list_positive, list_negative):
    ROOT_DIR = os.getcwd()

    # 训练集路径
    train_dir = os.path.join(ROOT_DIR, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_positive_dir = os.path.join(train_dir, "positive")
    train_negative_dir = os.path.join(train_dir, "negative")
    if not os.path.exists(train_negative_dir):
        os.mkdir(train_negative_dir)
    if not os.path.exists(train_positive_dir):
        os.mkdir(train_positive_dir)

    # 测试集路径
    test_dir = os.path.join(ROOT_DIR, "test")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    test_positive_dir = os.path.join(test_dir, "positive")
    test_negative_dir = os.path.join(test_dir, "negative")
    if not os.path.exists(test_negative_dir):
        os.mkdir(test_negative_dir)
    if not os.path.exists(test_positive_dir):
        os.mkdir(test_positive_dir)

    # 训练集：测试集 = 2:1
    # 构造训练集
    train_num = len(list_positive)//3 * 2
    for i in range(train_num):
        list_positive[i] = np.log2(list_positive[i] + 1)
        min = np.min(list_positive[i])
        max = np.max(list_positive[i])
        img_array = (list_positive[i] - min) / (max - min) * 255
        img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
        img.save(train_positive_dir + "//" + str(i) + ".png")

        list_negative[i] = np.log2(list_negative[i] + 1)
        min = np.min(list_negative[i])
        max = np.max(list_negative[i])
        img_array = (list_negative[i] - min) / (max - min) * 255
        img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
        img.save(train_negative_dir + "//" + str(i) + ".png")

    # 构造测试集
    for i in range(train_num, len(list_positive)):
        list_positive[i] = np.log2(list_positive[i] + 1)
        min = np.min(list_positive[i])
        max = np.max(list_positive[i])
        img_array = (list_positive[i] - min) / (max - min) * 255
        img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
        img.save(test_positive_dir + "//" + str(i) + ".png")

        list_negative[i] = np.log2(list_negative[i] + 1)
        min = np.min(list_negative[i])
        max = np.max(list_negative[i])
        img_array = (list_negative[i] - min) / (max - min) * 255
        img = Image.fromarray(np.array(img_array, dtype=int)).convert('RGB')
        img.save(test_negative_dir + "//" + str(i) + ".png")

def load_data():
    # 加载训练的正例样本
    # positive_samples = []
    # for i in range(1, 5):
    #     for j in range(1, 4):
    #         with open(rf'Positive_Sample_{i}_{j}.pkl', 'rb') as f:
    #             positive_samples.append(pickle.load(f))
    with open(rf'Positive_Sample_all.pkl', 'rb') as f:
        positive_samples = pickle.load(f)

    # 加载训练的负例样本
    # negative_samples = []
    # for i in range(1, 5):
    #     for j in range(1, 4):
    #         with open(rf'Negative_Sample_{i}_{j}.pkl', 'rb') as f:
    #             negative_samples.append(pickle.load(f))
    with open(rf'Negative_Sample_all.pkl', 'rb') as f:
        negative_samples = pickle.load(f)

    # print(len(positive_samples), len(negative_samples), type(positive_samples[0]), positive_samples[0].shape)

    # 构造训练集、测试集并保存为图片
    # save2image(positive_samples, negative_samples)
    save2image_new(positive_samples, negative_samples)

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=50,
                      param_group=False):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    print(devices)
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                  lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)

    # net = torch.load('model.pth')
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                       devices)
    torch.save(net, 'model_50.pth')

if __name__ == '__main__':
    # 构造数据集
    # load_data()

    data_dir = os.getcwd()

    # 使用RGB通道的均值和标准差，以标准化每个通道
    normalize = torchvision.transforms.Normalize(
        [0.485, ], [0.229, ])

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),  # 在原图像基础上修改，翻转概率为0.5
        torchvision.transforms.ToTensor(),
        normalize])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])

    finetune_net = torchvision.models.resnet18(pretrained=True)
    # finetune_net = torch.load('model_50.pth')
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    finetune_net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.xavier_uniform_(finetune_net.fc.weight);
    nn.init.xavier_uniform_(finetune_net.conv1.weight);

    print("begin")
    train_fine_tuning(finetune_net, 5e-2)
