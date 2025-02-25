# PyTorch 实战 Dog-Vs-Cat-Classification

可参考 [对应的markdown文件](./notes/note06_dog_vs_cat.md) 理解代码细节。

## 1 项目目录
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


## 2 数据下载

- 有关如何从 kaggle 下载的教程可见 [zhihu](https://zhuanlan.zhihu.com/p/25732245405)
- 解压后放入 AllData 文件下，或者自定义数据集的统一存放处【推荐】，文件目录大致为
```bash
AllData/
├── competitions
│   └── dog-vs-cat-classification
│       ├── test
│       │   └── test
│       │       ├── 000013.jpg
│       │       └── 000018.jpg
│       └── train
│           └── train
│               ├── cats
│               │   ├── cat.57.jpg
│               │   └── cat.62.jpg
│               └── dogs
│                   ├── dog.12.jpg
│                   └── dog.17.jpg
└── readme.md
```



## 3 安装

- PyTorch 的安装和环境配置可见 [zhihu](https://zhuanlan.zhihu.com/p/22230632892)
- 安装指定依赖：【进入 `requirements.txt` 根目录下安装】

```bash
pip install -r requirements.txt
```



## 4 训练

```bash
python main.py train
```

可以指定相关参数，参数写在 `config.py` 文件夹里，需要自己创建

```python
# config.py 在根目录下
import torch
import warnings

import os
from datetime import datetime


class DefaultConfig:
    model = 'AlexNetClassification'  # 选择模型
    root = './AllData/competitions/dog-vs-cat-classification'  # 填入数据集位置

    # 获取最新的文件
    param_path = './checkpoints/'  # 存放模型位置
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

    tensorboard_log_dir = './logs'  # 存放 Tensorboard 的 logs 文件

    result_file = 'result.csv'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        config.device = torch.device('cuda') if config.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


config = DefaultConfig()
```

可以在命令后中修改

```bash
python main.py train --root=/Users/...
```



## 5 测试

```
python main.py test
```

然后在根目录下会得到 `result.csv` 文件，可以上传到 kaggle



## 6 友链

1. 关注我的知乎账号 [Zhuhu](https://www.zhihu.com/people/--55-97-8-41) 不错过我的笔记更新。
2. 我会在个人博客 [isKage`Blog](https://blog.iskage.online/) 更新相关项目和学习资料。

