# 搭建神经网络 PyTorch 实战: 猫狗分类问题训练全流程

根据之前的知识 [专栏：PyTorch 教程](https://zhuanlan.zhihu.com/column/c_1864780737208799232)，本文将一步一步详细介绍如何获取数据，自定义数据集，搭建神经网络，训练、验证和测试结果的全流程。针对的问题是 kaggle 经典竞赛：猫狗二分类问题（[dog-vs-cat-classification](https://www.kaggle.com/competitions/dog-vs-cat-classification)）

相关代码均已开源在我的 Github 库 [https://github.com/isKage/dog-vs-cat-classification](https://github.com/isKage/dog-vs-cat-classification) 。【注意】代码只是用于介绍搭建网络、训练模型，实际参考网络效果极差，不建议使用。文章根据 [深度学习框架PyTorch：入门与实践 (陈云) ](https://github.com/chenyuntc/pytorch-book) 整理。

注意数据集较大，如果想要训练出结果需要 GPU。如果只是学习如何训练搭建网络等流程，可以自己在根目录下创建 `AllData` 文件夹，模拟数据集文件结构放入部分图片。

如果想直接运行我的 Github 代码，还可以阅读 [README.md](https://github.com/isKage/dog-vs-cat-classification/blob/main/README.md) 文档。

## 0 项目目录

如果直接下载我的 Github 源码。进入一个空目录后，打开终端输入【或者跟着后文一步一步编写】

```bash
git clone https://github.com/isKage/dog-vs-cat-classification.git
```

目录结构

```bash
.
├── AllData  # 数据集存放
├── README.md
├── checkpoints  # 训练好的模型        【需要自己创建】
├── config.py  # 配置文件，如何创建见下  【需要自己创建】
├── data  # 自定义数据集处理包
│   ├── __init__.py
│   │   └── dataset.cpython-312.pyc
│   └── dataset.py
├── logs  # 存放 tensorboard logs 文件 【需要自己创建】
├── main.py  # 主程序
├── models  # 网络模型定义
│   ├── __init__.py
│   ├── basic.py
│   └── cnn.py
├── notes  # 一些笔记
│   ├── kaggle_download.md
│   └── note06_dog_vs_cat.md
├── requirements.txt  # 依赖包
├── result.csv  # 预测/测试结果
└── utils  # 一些辅助包
    ├── __init__.py
    └── visualizer.py  # 封装可视化功能
```

安装依赖

```bash
pip install -r requirements.txt
```



## 1 获取数据

从 kaggle 官网下载数据集，可以自定义数据集的统一放置路径，方便未来训练使用。具体教程见：[从 Kaggle 下载数据集（mac 和 win 端）](https://zhuanlan.zhihu.com/p/25732245405)

下载后解压，数据集结构大致为：

```bash
dog-vs-cat-classification
├── dog-vs-cat-classification.zip
├── sample_submission.csv
├── test
│   └── test # 测试图片
└── train
    └── train  # 训练图片
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1740458311037.png)



## 2 自定义数据集

根据如上的数据集文件结构，我们自定义数据集 Dataset 类。相关教程见：[数据处理：Dataset 类和 Dataloader 类](https://zhuanlan.zhihu.com/p/23210343084)。

定义 Python 函数包 `data` ，结构如下

```bash
data
├── __init__.py  # 初始化包
└── dataset.py  # 自定义的 dataset 类
```

- 使用 `__init__.py` 声明 `data` 文件夹视为 Python 的程序包

```python
# __init__.py
from .dataset import DogVsCatDataset
```

其中 `.dataset` 指向文件 `dataset.py` 而 `DogVsCatDataset` 为我们在 `dataset.py` 中自定义的类

如此就可以在其他程序里使用 

```python
from data import DogVsCatDataset
```

- `dataset.py` 中定义读取猫狗分类问题的数据集 Dataset 类

```python
# dataset.py
import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DogVsCatDataset(Dataset):
    """加载猫狗数据集"""

    def __init__(self, root, trans=None, mode=None):
        """
        初始化
        :param root: 数据集文件路径
        :param trans: 变换操作
        :param mode: ['train', 'val', 'test']
        """
        assert mode in ['train', 'val', 'test']  # 判断 mode 是否合法，否则报错
        self.mode = mode

        if self.mode != 'test':
            # 训练集和验证集要把猫狗训练数据都获取
            root = os.path.join(root, 'train', 'train')
            img_dir_dict = [os.path.join(root, 'cats', img_dir) for img_dir in os.listdir(os.path.join(root, 'cats'))]
            img_dir_dict += [os.path.join(root, 'dogs', img_dir) for img_dir in os.listdir(os.path.join(root, 'dogs'))]
            random.shuffle(img_dir_dict)  # 猫狗图片打乱
        else:
            # 测试集路径不同
            root = os.path.join(root, 'test', 'test')
            img_dir_dict = [os.path.join(root, img_dir) for img_dir in os.listdir(os.path.join(root))]

        img_num = len(img_dir_dict)

        # 存入图片路径
        if self.mode == 'test':
            self.img_dir_dict = img_dir_dict
        # 划分数据集
        elif self.mode == 'train':
            self.img_dir_dict = img_dir_dict[:int(img_num * 0.7)]
        else:
            self.img_dir_dict = img_dir_dict[int(img_num * 0.7):]

        if trans is None:
            # 数据转换操作，测试、验证和训练集的数据转换有所区别
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])

            # 测试集 test 和验证集 val 不需要数据增强
            if self.mode == "test" or self.mode == "val":
                self.trans = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])
            # 训练集 需要数据增强
            else:
                self.trans = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        真正开始读取数据，对于测试集 test 返回 id，如 100.jpg 返回 100
        :param index: 图片下标
        :return: 返回张量数据和标签
        """
        img_path = self.img_dir_dict[index]
        if self.mode == "test":
            label = int(os.path.basename(img_path).split('.')[0])
        else:
            # dog is 1, cat is 0
            label = 1 if 'dog' in os.path.basename(img_path).split('.') else 0

        # 读取图片
        data = Image.open(img_path)
        data = self.trans(data)
        return data, label

    def __len__(self):
        """
        返回图片个数
        :return: 数据集大小
        """
        return len(self.img_dir_dict)

if __name__ == '__main__':
    root = "../AllData/competitions/dog-vs-cat-classification"
    train_dataset = DogVsCatDataset(root, mode='train')
    test_dataset = DogVsCatDataset(root, mode='test')
    print(len(train_dataset))
    print(len(test_dataset))

    print(os.path.basename(train_dataset.img_dir_dict[0]).split('.'))
    print(os.path.basename(test_dataset.img_dir_dict[0]).split('.')[0])
```



## 3 搭建网络模型

创建 Python 包 `models` 

```bash
models
├── __init__.py
├── basic.py
└── cnn.py
```

- 同样地，使用 `__init__.py` 声明

```python
# __init__.py
from .cnn import AlexNetClassification
```

- 为了方便的保存模型和加载模型，我们定义一个基类，继承 `nn.Module` 提前添加 `save` 和 `load` 方法

```python
# basic.py
import time
import torch

from torch import nn


class BasicModule(nn.Module):
    """
    作为基类，继承 nn.Module 但增加了模型保存和加载功能 save and load
    """

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        """
        根据模型路径加载模型
        :param model_path: 模型路径
        :return: 模型
        """
        self.load_state_dict(torch.load(model_path))

    def save(self, filename=None):
        """
        保存模型，默认使用 "模型名字 + 时间" 作为文件名，也可以自定义
        """
        if filename is None:
            filename = 'checkpoints/' + self.model_name + '_' + time.strftime("%Y-%m-%d%H%M%S") + '.pth'
        torch.save(self.state_dict(), filename)
        return filename
```

- 然后定义真正的网络模型，我们这使用了最传统简单的 AlexNet 网络【故效果不佳，可以前往 kaggle 查看更好的模型】，神经网络搭建教程可见：[利用 torch.nn 搭建神经网络](https://zhuanlan.zhihu.com/p/22793450207)

```python
# cnn.py
import torch
from torch import nn

from .basic import BasicModule


class AlexNetClassification(BasicModule):
    def __init__(self, num_classes=2):
        super(AlexNetClassification, self).__init__()
        self.model_name = 'CNNClassification'

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```



## 4 可视化插件

为了方便直观地观测训练过程，我们可以提前自定义一个可视化的 Python 包，如上一样的套路

```bash
utils
├── __init__.py
└── visualizer.py
```

- `__init__py`  声明

```python
# __init__.py
from .visualizer import Visualizer
```

- `visualizer.py` 中封装可视化的类，这里使用的是 Tensorboard ，具体教程可见：[可视化工具：Tensorboard](https://zhuanlan.zhihu.com/p/23467081773)

```python
import time
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    封装了基本的 TensorBoard 操作。
    """

    def __init__(self, log_dir):
        # 初始化 TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.index = {}  # 用于追踪图表的点
        self.log_text = ''  # 用于记录日志信息

    def reinit(self, log_dir, **kwargs):
        """
        重新初始化 TensorBoard writer，并设置新的日志目录。
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        return self

    def plot(self, name, y, step=None):
        """
        将标量值记录到 TensorBoard。
        例如：plot('loss', 1.00)
        """
        if step is None:
            step = self.index.get(name, 0)
        self.writer.add_scalar(name, y, step)
        self.index[name] = step + 1

    def img(self, name, img_, step=None):
        """
        将图像记录到 TensorBoard。
        img_ 应该是一个张量（例如，torch.Tensor）。
        """
        if step is None:
            step = self.index.get(name, 0)
        self.writer.add_images(name, img_, step)

    def log(self, info, step=None, win='log_text'):
        """
        记录信息为文本（可选）。
        """
        if step is None:
            step = self.index.get(win, 0)
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.writer.add_text(win, self.log_text, step)
        self.index[win] = step + 1

    def __getattr__(self, name):
        """
        允许访问其他 TensorBoard writer 的函数。
        """
        return getattr(self.writer, name)
```

> 使用 tensorboard ：下载 tensorboard ，程序完成后使用命令开启

```bash
pip install tensorboard
```

```bash
tensorboard --logdir=./logs
```

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/QQ_1740458248881.png)



## 5 config 本地配置和 fire 包

### 5.1 config 本地配置

创建 `config.py` 用于提取写好训练的参数

```python
# config.py
import torch
import warnings

import os
from datetime import datetime


class DefaultConfig:
    model = 'AlexNetClassification'
    root = './AllData/competitions/dog-vs-cat-classification'

    # 获取最新的文件
    param_path = './checkpoints/'
    if not os.listdir(param_path):
        load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    else:
        load_model_path = os.path.join(
            param_path,
            sorted(
                os.listdir(param_path),
                key=lambda x: datetime.strptime(
                    x.split('_')[-1].split('.pth')[0],
                    "%Y-%m-%d%H%M%S"
                )
            )[-1]
        )

    batch_size = 32
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    num_workers = 0
    print_freq = 20

    max_epochs = 10
    lr = 0.003
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    tensorboard_log_dir = './logs'

    result_file = 'result.csv'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        config.device = torch.device('cuda:0') if config.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


config = DefaultConfig()
```

### 5.2 fire 包

`fire` 包能够在终端直接进行传参和训练。先下载

```bash
pip install fire
```

而后当主程序 `main.py` 中定义了训练、测试、验证函数时

```python
# main.py
from config import config

def train(**kwargs):
    ...
    
@torch.no_grad()
def val(**kwargs):
    ...
    
@torch.no_grad()
def test(**kwargs):
    ...
```

于是可以在终端中使用如下方法进行训练和传入参数。

```bash
python main.py train --参数1=<param> --参数2=<param>
```



## 6 主程序

### 6.1 训练和验证函数

```python
# main.py
from config import config
from data import DogVsCatDataset
from torch.utils.data import DataLoader
import models
import torch
from tqdm import tqdm
from utils import Visualizer


def train(**kwargs):
    # 根据命令行参数更新配置
    config._parse(kwargs)
    vis = Visualizer(log_dir=config.tensorboard_log_dir)  # 使用 TensorBoard

    # step1: 模型
    model = getattr(models, config.model)()
    model.to(config.device)

    # step2: 数据
    train_data = DogVsCatDataset(config.root, mode="train")
    val_data = DogVsCatDataset(config.root, mode="val")
    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, config.batch_size, shuffle=False, num_workers=config.num_workers)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # 初始化误差
    previous_loss = 1e10

    # 训练
    for epoch in range(config.max_epochs):
        epoch_loss = 0  # 记录当前 epoch 的平均损失

        for ii, (data, label) in enumerate(train_dataloader):
            # 训练模型参数
            inputs = data.to(config.device)
            target = label.to(config.device)

            optimizer.zero_grad()
            score = model(inputs)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 记录损失到 TensorBoard
            if (ii + 1) % config.print_freq == 0:
                vis.plot('loss', loss.item(), step=epoch * len(train_dataloader) + ii)

        # 保存模型
        model.save()

        # 在每个 epoch 结束后验证模型
        val_accuracy = val(model, val_dataloader)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        vis.plot('val_accuracy', val_accuracy, step=epoch)

        # 记录训练日志
        vis.log(
            f"epoch:{epoch}, lr:{lr}, loss:{epoch_loss / len(train_dataloader):.4f}, val_accuracy:{val_accuracy:.4f}"
        )

        # 更新学习率
        if epoch_loss / len(train_dataloader) > previous_loss:
            lr = lr * config.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = epoch_loss / len(train_dataloader)


@torch.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    correct = 0
    total = 0
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(config.device)
        label = label.to(config.device)
        score = model(val_input)
        _, predicted = score.max(1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    model.train()
    accuracy = 100. * correct / total
    return accuracy
```

### 6.2 测试和写入结果函数

```python
# main.py
@torch.no_grad()
def test(**kwargs):
    config._parse(kwargs)

    # configure model
    model = getattr(models, config.model)().eval()
    if config.load_model_path:
        model.load(config.load_model_path)

    model.to(config.device)

    # data
    test_data = DogVsCatDataset(config.root, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        inputs = data.to(config.device)
        score = model(inputs)

        predicted_label = score.max(dim=1)[1].detach().tolist()

        # 如果你要保存为 id, label 的格式，修改为：
        batch_results = [(path_.item(), label_) for path_, label_ in zip(path, predicted_label)]

        results += batch_results

    write_csv(results, config.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)
```

最后使用 fire 包

```python
if __name__ == '__main__':
    import fire

    fire.Fire()
```



## 7 训练和预测过程

### 7.1 开始训练

```bash
python main.py train
```

一定要先定义好参数和数据集路径。

![](https://blog-iskage.oss-cn-hangzhou.aliyuncs.com/images/47ef5337feb656be8b726fe2cdcf8f6f_720.png)

### 7.2 测试

在上面训练结束后

```bash
python main.py test
```

### 7.3 结果

得到结果，写入了 `result.csv` ，之后可以提交到 kaggle













