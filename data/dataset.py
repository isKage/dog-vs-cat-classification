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
